# -*- coding: utf-8 -*-
"""
模块说明: Transformer 编码器与解码器核心堆叠层 (layers/Transformer_EncDec.py)
-------------------------------------------------------------------------
该模块定义了标准时序 Transformer 模型中的基础网络模块：
1. `ConvLayer`: 蒸馏下采样卷积层，通过 1D 卷积与最大池化使时序序列长度减半 (L -> L/2)，提升计算效率。
2. `EncoderLayer`: 单个编码器层，包含多头自注意力 (Self-Attention)、前馈网络 (FFN)、残差连接与层归一化。
3. `Encoder`: 多层编码器容器，支持深层特征抽取与跨层蒸馏。
4. `DecoderLayer`: 单个解码器层，包含因果掩码自注意力、与编码器输出交互的交叉注意力 (Cross-Attention) 以及 FFN。
5. `Decoder`: 多层解码器容器与线性投影头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    时序特征蒸馏下采样层 (Informer Distillation Layer)

    通过一维循环卷积 (Circular Padding Conv1d) 与步长为 2 的最大池化 (MaxPool1d)，
    将时序序列长度减半，提取最显著的时序模式并大幅节约显存与计算量。

    张量流动与变换追踪:
        输入 x: [B, L, C]
        permute(0, 2, 1): [B, C, L]
        Conv1d (padding=2, circular) + BatchNorm + ELU + MaxPool1d(stride=2): [B, C, L // 2]
        transpose(1, 2): [B, L // 2, C]

    Args:
        c_in (int): 输入与输出的通道特征维度
    """

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        # 循环填充卷积保持周期边缘连续性
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        # stride=2 的池化操作将序列长度压缩至原有的一半
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入时序特征，形状为 [B, L, C]

        Returns:
            torch.Tensor: 下采样后的时序特征，形状为 [B, L // 2, C]
        """
        # [B, L, C] -> 转置为通道在前 -> [B, C, L]
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # [B, C, L // 2]
        # 恢复为 [B, L // 2, C]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    单个 Transformer 编码器层 (Encoder Layer)

    由多头注意力机制模块、逐位置前馈网络 (Position-wise Feed-Forward Network)、
    两次残差连接 (Residual Connections) 以及层归一化 (LayerNorm) 构成。

    Args:
        attention (nn.Module): 注意力计算子层 (如 AttentionLayer(FullAttention))
        d_model (int): 隐藏状态通道特征维度
        d_ff (int, optional): 前馈网络隐藏层扩展维度. 默认为 4 * d_model.
        dropout (float, optional): Dropout 失活概率. 默认为 0.1.
        activation (str, optional): 激活函数 ('relu' 或 'gelu'). 默认为 "relu".
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # 使用 1D 逐点卷积 (kernel_size=1) 实现前馈网络 (FFN)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        张量流动追踪:
            输入 x: [B, L, d_model]
            注意力子层: [B, L, d_model] -> new_x: [B, L, d_model]
            残差与归一化 1: x = LayerNorm(x + Dropout(new_x)) -> [B, L, d_model]
            FFN 子层: [B, L, d_model] -> Conv1d -> [B, d_ff, L] -> Conv1d -> [B, L, d_model]
            残差与归一化 2: out = LayerNorm(x + Dropout(y)) -> [B, L, d_model]

        Args:
            x (torch.Tensor): 输入时序特征张量 [B, L, d_model]
            attn_mask (torch.Tensor, optional): 自注意力掩码
            tau (float, optional): 平稳注意力缩放因子
            delta (float, optional): 时序偏移量

        Returns:
            torch.Tensor: 编码器层输出 [B, L, d_model]
            attn (torch.Tensor, optional): 注意力权重图矩阵
        """
        # 1. 自注意力计算与第一个残差连接
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # 2. 前馈神经网络与第二个残差连接
        # y: [B, L, d_model] -> transpose -> [B, d_model, L] -> conv1 -> [B, d_ff, L]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y: [B, d_ff, L] -> conv2 -> [B, d_model, L] -> transpose -> [B, L, d_model]
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    多层编码器容器 (Encoder Container)

    管理并串联多个 EncoderLayer，同时可选在层间插入 ConvLayer 进行时序下采样。

    Args:
        attn_layers (list of nn.Module): 编码器层列表
        conv_layers (list of nn.Module, optional): 下采样卷积层列表 (长度通常为 len(attn_layers) - 1)
        norm_layer (nn.Module, optional): 最终输出的全局层归一化 (LayerNorm)
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        张量流动追踪:
            x 输入: [B, L, d_model]
            逐层经过 EncoderLayer (+ ConvLayer): 序列长度保持 L 或递减 (L -> L/2 -> ...)
            输出: [B, L_out, d_model]

        Args:
            x (torch.Tensor): 输入时序张量 [B, L, d_model]
            attn_mask (torch.Tensor, optional): 掩码张量
            tau (float, optional): 温度调节参数
            delta (float, optional): 偏移参数

        Returns:
            x (torch.Tensor): 编码完成的高级表征 [B, L_out, d_model]
            attns (list): 各层返回的注意力权重图
        """
        attns = []
        if self.conv_layers is not None:
            # 包含层间蒸馏下采样模式
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta_i = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_i)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # 标准全长堆叠模式 (序列长度 L 保持不变)
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    单个 Transformer 解码器层 (Decoder Layer)

    包含三大核心子层:
    1. 掩码自注意力 (Masked Self-Attention): 限制当前时刻仅能关注历史信息。
    2. 交叉注意力 (Cross-Attention): 以解码器特征为 Query，与编码器输出 (Key, Value) 交互。
    3. 前馈神经网络 (FFN) + 残差连接与归一化。

    Args:
        self_attention (nn.Module): 掩码自注意力模块
        cross_attention (nn.Module): 跨模态/跨层交叉注意力模块
        d_model (int): 隐藏层通道维度
        d_ff (int, optional): 前馈网络中间通道维度. 默认为 4 * d_model.
        dropout (float, optional): Dropout 丢弃率. 默认为 0.1.
        activation (str, optional): 激活函数 ('relu' 或 'gelu'). 默认为 "relu".
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        张量流动追踪:
            x: [B, L_dec, d_model]
            cross (编码器输出): [B, L_enc, d_model]
            输出: [B, L_dec, d_model]
        """
        # 1. 解码器自注意力 + 残差连接 1
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # 2. 与编码器表征的交叉注意力 (Q=x, K=cross, V=cross) + 残差连接 2
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x = self.norm2(x)

        # 3. 逐位置前馈网络 + 残差连接 3
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    多层解码器容器 (Decoder Container)

    串联堆叠多个 DecoderLayer，并可选接线性映射层投影至目标维度 (c_out)。

    Args:
        layers (list of nn.Module): 解码器层列表
        norm_layer (nn.Module, optional): 全局层归一化
        projection (nn.Module, optional): 最终线性投影头 (nn.Linear(d_model, c_out))
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Args:
            x (torch.Tensor): 解码器输入张量 [B, L_dec, d_model]
            cross (torch.Tensor): 编码器输出高级表征 [B, L_enc, d_model]
            x_mask (torch.Tensor, optional): 自注意力因果掩码
            cross_mask (torch.Tensor, optional): 交叉注意力掩码

        Returns:
            torch.Tensor: 解码预测输出，若有 projection 则为 [B, L_dec, c_out]，否则为 [B, L_dec, d_model]
        """
        # 逐层解码计算
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x