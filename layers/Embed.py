# -*- coding: utf-8 -*-
"""
模块说明: 时序多特征融合嵌入层模块 (layers/Embed.py)
-------------------------------------------------------------------------
该模块为时序 Transformer 模型提供了全套特征嵌入解决方案，主要包含：
1. `TokenEmbedding`: 数值特征嵌入，通过 1D 循环填充卷积将原始时序输入通道 (c_in) 映射至隐藏层通道 (d_model)。
2. `PositionalEmbedding`: 经典正余弦绝对位置编码，赋予序列位置秩序感知能力。
3. `TemporalEmbedding` / `TimeFeatureEmbedding`: 时间特征嵌入，将离散时间戳（月、日、周几、小时）编码为密集向量。
4. `DataEmbedding` 族: 组合数值嵌入、位置嵌入与时间嵌入，并支持不同嵌入组件的消融变体。
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compared_version(ver1, ver2):
    """
    版本号对比辅助函数

    Args:
        ver1 (str): 当前版本号 (如 torch.__version__)
        ver2 (str): 目标参考版本号 (如 '1.5.0')

    Returns:
        int or bool: 比较结果 (-1, 0, 1 或布尔值)
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")

    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        try:
            v1_i = int(list1[i].split('+')[0])
            v2_i = int(list2[i].split('+')[0])
            if v1_i < v2_i:
                return -1
            elif v1_i > v2_i:
                return 1
        except ValueError:
            pass

    return len(list1) >= len(list2)


class PositionalEmbedding(nn.Module):
    """
    正余弦绝对位置编码 (Sinusoidal Positional Embedding)

    预先计算固定位置的正弦与余弦响应，并在训练时直接切片提取，无需梯度反向传播。

    Args:
        d_model (int): 嵌入特征维度
        max_len (int, optional): 最大支持时序序列长度. 默认为 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        # 生成位置索引: [0, 1, ..., max_len - 1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 频率分母因子: 10000 ^ (2i / d_model)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 偶数索引使用正弦，奇数索引使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展出 Batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        根据输入序列长度截取对应位置编码

        Args:
            x (torch.Tensor): 输入时序张量，第二维为序列长度 L，形状为 [B, L, ...]

        Returns:
            torch.Tensor: 截取的位置编码张量，形状为 [1, L, d_model]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    时序数值特征卷积嵌入层 (Token Embedding)

    通过带有 circular padding 的 1D 卷积，将输入原始特征通道数 c_in 投影到 d_model 维度，
    并提取局部时序上下文信息。

    张量流动与变换追踪:
        输入 x: [B, L, c_in]
        permute(0, 2, 1): [B, c_in, L]
        tokenConv 1D 卷积: [B, d_model, L]
        transpose(1, 2): [B, L, d_model]

    Args:
        c_in (int): 原始数值特征维度 (如 24 小时通道)
        d_model (int): 投影目标隐藏层维度 (如 64)
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据 PyTorch 版本选择 padding 尺寸
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        # Kaiming 正态分布初始化卷积核权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量 [B, L, c_in]

        Returns:
            torch.Tensor: 嵌入张量 [B, L, d_model]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    固定权重正余弦查表嵌入 (Fixed Embedding)

    将离散索引通过固定的正余弦函数映射至密集表征，梯度不更新 (requires_grad=False)。
    """

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    离散时间戳多粒度嵌入层 (Temporal Embedding)

    将时间标记中的月份、日、周几、小时分别编码为嵌入向量后逐元素相加求和。

    Args:
        d_model (int): 嵌入特征维度
        embed_type (str, optional): 嵌入类型 ('fixed' 固定正余弦编码, 'learned' 可学习嵌入). 默认为 'fixed'.
        freq (str, optional): 时间频率. 默认为 'h' (小时级).
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        hour_size = 24       # 每天 24 小时
        weekday_size = 7     # 每周 7 天
        day_size = 182       # 序列最大跨越 182 天
        month_size = 6       # 跨越月份数

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(4, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 时间戳特征矩阵，形状为 [B, L, 4]，通道对应 (月, 日, 星期, 小时)

        Returns:
            torch.Tensor: 融合多粒度时间戳信息后的张量 [B, L, d_model]
        """
        x = x.long()
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return weekday_x + day_x + month_x + hour_x


class TimeFeatureEmbedding(nn.Module):
    """
    连续时间特征线性嵌入层 (Time Feature Linear Embedding)

    适用于输入为经过归一化的连续时间标量矩阵 (如由 pandas DateOffset 提取的特征)。
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# =========================================================================
# 多模态复合数据嵌入组合 (DataEmbedding Classes)
# =========================================================================

class DataEmbedding(nn.Module):
    """
    完整时序数据嵌入模块:
    输出 = TokenEmbedding(数值) + PositionalEmbedding(位置) + TemporalEmbedding(时间)

    张量流动与变换追踪:
        x: [B, L, c_in] -> TokenEmbedding -> [B, L, d_model]
        x: [B, L, c_in] -> PositionalEmbedding -> [1, L, d_model]
        x_mark: [B, L, d_time] -> TemporalEmbedding -> [B, L, d_model]
        总和: [B, L, d_model]
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """消融变体: 无位置编码嵌入 (Token + Temporal)"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != 'timeF'
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """消融变体: 纯数值卷积嵌入 (仅 Token，无位置且无时间)"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    """消融变体: 无时间标记嵌入 (Token + Position)"""

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)