# -*- coding: utf-8 -*-
"""
模块说明: 标准时序 Transformer 编码器-解码器模型架构 (Transformer.py)
-------------------------------------------------------------------------
该模块实现了基于自注意力机制 (Self-Attention) 的经典时序 Transformer 模型架构，
用于对人群到访时空数据的长序列时间动态进行深度特征抽取与编码。

包含核心组件:
1. 多类型时序嵌入层 (DataEmbedding): 支持数值 Token 嵌入、位置编码 (Position) 与时间标记编码 (Temporal)。
2. 多层堆叠式编码器 (Encoder / EncoderLayer): 基于 FullAttention 计算全时序范围内的注意力相关性。
3. 多层堆叠式解码器 (Decoder / DecoderLayer): 支持自回归因果掩码注意力与交叉注意力。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_temp,
    DataEmbedding_wo_pos_temp
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    经典时序 Transformer 架构 (复杂度 O(L^2))

    Args:
        config (dict): 模型配置参数字典，包含以下主要字段:
            - 'pred_len' (int): 预测时序步长 / 输出序列长度.
            - 'output_attention' (bool): 是否返回注意力热力图矩阵.
            - 'enc_in' (int): 编码器输入特征通道数 (如到访数据每小时特征 24).
            - 'dec_in' (int): 解码器输入特征通道数.
            - 'c_out' (int): 输出目标特征通道数.
            - 'd_model' (int): Transformer 隐藏层通道维度 (如 64).
            - 'n_heads' (int): 多头注意力的头数 (如 4 或 8).
            - 'e_layers' (int): 编码器层数 (如 2).
            - 'd_layers' (int): 解码器层数 (如 1).
            - 'd_ff' (int): 前馈神经网络隐藏层扩展维度 (如 512).
            - 'dropout' (float): Dropout 随机失活概率.
            - 'activation' (str): 激活函数类型 ('relu' 或 'gelu').
            - 'factor' (int): 注意力采样因子.
            - 'embed_type' (int): 嵌入类型编码:
                0: 完整嵌入 (Token + Pos + Temp)
                1: 完整嵌入
                2: 无位置编码 (wo_pos)
                3: 无时间标记编码 (wo_temp)
                4: 无位置且无时间编码 (wo_pos_temp)
            - 'freq' (str): 时间标记频率 ('h', 't', 's' 等).
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.pred_len = config['pred_len']
        self.output_attention = config['output_attention']

        # ------------------ 1. 时序嵌入层 (Embedding) ------------------
        # 根据 embed_type 选择对应的时间/位置/数值组合嵌入方式
        if config['embed_type'] in [0, 1]:
            # 完整嵌入: 数值线性映射 + 正余弦绝对位置编码 + 周期时间特征编码
            self.enc_embedding = DataEmbedding(
                config['enc_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
            self.dec_embedding = DataEmbedding(
                config['dec_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
        elif config['embed_type'] == 2:
            # 仅包含数值与时间标记编码，剔除绝对位置编码
            self.enc_embedding = DataEmbedding_wo_pos(
                config['enc_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                config['dec_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
        elif config['embed_type'] == 3:
            # 仅包含数值与位置编码，剔除时间标记编码
            self.enc_embedding = DataEmbedding_wo_temp(
                config['enc_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
            self.dec_embedding = DataEmbedding_wo_temp(
                config['dec_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
        elif config['embed_type'] == 4:
            # 仅包含数值线性投影，无位置和时间编码
            self.enc_embedding = DataEmbedding_wo_pos_temp(
                config['enc_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )
            self.dec_embedding = DataEmbedding_wo_pos_temp(
                config['dec_in'], config['d_model'], config['embed'], config['freq'], config['dropout']
            )

        # ------------------ 2. 编码器 (Encoder) ------------------
        # 由多个 EncoderLayer 串联构成
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=config['factor'],
                            attention_dropout=config['dropout'],
                            output_attention=config['output_attention']
                        ),
                        d_model=config['d_model'],
                        n_heads=config['n_heads']
                    ),
                    d_model=config['d_model'],
                    d_ff=config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation']
                ) for _ in range(config['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model'])
        )

        # ------------------ 3. 解码器 (Decoder) ------------------
        # 包含自注意力和与编码器特征交互的交叉注意力
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,  # 自回归因果掩码
                            factor=config['factor'],
                            attention_dropout=config['dropout'],
                            output_attention=False
                        ),
                        d_model=config['d_model'],
                        n_heads=config['n_heads']
                    ),
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=config['factor'],
                            attention_dropout=config['dropout'],
                            output_attention=False
                        ),
                        d_model=config['d_model'],
                        n_heads=config['n_heads']
                    ),
                    d_model=config['d_model'],
                    d_ff=config['d_ff'],
                    dropout=config['dropout'],
                    activation=config['activation'],
                )
                for _ in range(config['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(config['d_model']),
            projection=nn.Linear(config['d_model'], config['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Transformer 前向推理与特征抽取

        张量流动追踪:
            x_enc: [B, L_in, enc_in] (如 [B, 182, 24])
            x_mark_enc: [B, L_in, 4]
            enc_out: [B, L_in, d_model] (如 [B, 182, 64])
            output: [B, pred_len, d_model]

        Args:
            x_enc (torch.Tensor): 编码器输入序列，形状为 [B, L_in, enc_in]
            x_mark_enc (torch.Tensor, optional): 编码器时间标记张量，形状为 [B, L_in, 4]
            x_dec (torch.Tensor, optional): 解码器输入序列，形状为 [B, L_out, dec_in]
            x_mark_dec (torch.Tensor, optional): 解码器时间标记张量，形状为 [B, L_out, 4]
            enc_self_mask (torch.Tensor, optional): 编码器自注意力掩码
            dec_self_mask (torch.Tensor, optional): 解码器自注意力因果掩码
            dec_enc_mask (torch.Tensor, optional): 解码器交叉注意力掩码

        Returns:
            torch.Tensor: 编码器抽取的高级时序表征，形状为 [B, pred_len, d_model]
            attns (list, optional): 注意力分布图列表 (当 output_attention=True 时)
        """
        # 获取当前实际 Batch Size (B) 与序列长度
        b, l, _ = x_enc.size()

        # 若未提供时间戳标记，使用占位张量
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(b, l, 4, device=x_enc.device)

        # 1. 编码器特征嵌入: 数值映射 + 位置编码 + 时间编码
        # x_enc: [B, 182, 24] -> enc_embedding -> enc_out: [B, 182, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 2. 编码器层前向计算 (多头自注意力 + 前馈神经网络 + 残差连接 + LayerNorm)
        # enc_out: [B, 182, d_model] -> encoder -> enc_out: [B, 182, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # 3. 输出截取: 截取最后 pred_len 步的时序表征
        # enc_out: [B, 182, d_model] -> [B, pred_len, d_model]
        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]