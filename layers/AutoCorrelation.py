# -*- coding: utf-8 -*-
"""
模块说明: 自相关时序关联机制模块 (layers/AutoCorrelation.py)
-------------------------------------------------------------------------
该模块实现了 Autoformer (NeurIPS 2021) 提出的自相关机制 (Auto-Correlation Mechanism)，
用于无缝替代传统的逐点自注意力 (Self-Attention) 机制：
1. 周期相关性发现 (Period-based dependencies discovery):
   - 基于维纳-辛钦定理 (Wiener-Khinchin Theorem)，在频域内利用快速傅里叶变换 (FFT) 计算序列的自相关系数。
   - 计算复杂度仅为 O(L log L)，能跨越长时序直接捕获子序列级别的周期性相似模式。
2. 时延信息聚合 (Time delay aggregation):
   - 提取自相关系数最大的 Top-k 周期时延 (Time Delays)，通过滚动平移 (Roll) 与相干加权实现长时序表征融合。
"""

import math
from math import sqrt
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """
    自相关注意力机制计算核心模块 (AutoCorrelation Core Module)

    通过频域卷积与时延滚动加权，直接在子序列级别发现周期相似性，并实现信息聚合。

    Args:
        mask_flag (bool, optional): 掩码标记 (AutoCorrelation 原生支持无掩码周期建模). 默认为 True.
        factor (int, optional): Top-k 周期选择缩放因子. 默认为 1.
        scale (float, optional): 缩放系数. 默认为 None.
        attention_dropout (float, optional): Dropout 概率. 默认为 0.1.
        output_attention (bool, optional): 是否返回自相关矩阵. 默认为 False.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        训练阶段快速时延聚合函数 (利用 torch.roll 批次平移)

        张量流动追踪:
            values: [B, H, C, L] (通道前置格式)
            corr: [B, H, C, L] (自相关系数矩阵)
            delays_agg 输出: [B, H, C, L]
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # 1. 计算选取的最大自相关时延数量: top_k = c * ln(L)
        top_k = int(self.factor * math.log(length))

        # 2. 沿通道与多头求均值，挑选全局能量最显著的 Top-k 时延索引
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  # [top_k]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # [B, top_k]

        # 3. Softmax 归一化时延权重
        tmp_corr = torch.softmax(weights, dim=-1)

        # 4. 根据时延索引循环滚动 values 并加权求和
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # 将序列沿时间维平移 index[i] 步长
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        推断阶段快速时延聚合函数 (基于 torch.gather 索引提取)
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # 构造基准索引张量
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)

        # 选取每个样本专属的 Top-k 时延
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # [B, L]
        weights, delay = torch.topk(mean_value, top_k, dim=-1)   # [B, top_k]

        tmp_corr = torch.softmax(weights, dim=-1)

        # 将序列拼接两份以模拟周期循环采样: [B, H, C, 2*L]
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """标准完整版时延聚合实现 (细粒度保留多头通道维度)"""
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)

        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)

        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        """
        张量流动与变换追踪:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            values: [B, S, H, D]
            频域 FFT:
                rfft(queries.permute(0, 2, 3, 1)) -> q_fft: [B, H, E, L//2 + 1] (复数张量)
                q_fft * conj(k_fft) -> 互功率谱 (Cross-Power Spectrum)
                irfft(...) -> 逆变换恢复时域自相关矩阵 corr: [B, H, E, L]
            时延聚合输出 V: [B, L, H, D]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # 序列长度对齐
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # 1. 周期相关性计算 (维纳-辛钦定理频域卷积)
        # 转置为 [B, H, E, L]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # 频域共轭内积
        res = q_fft * torch.conj(k_fft)
        # 逆傅里叶变换生成全序列时延自相关系数 corr: [B, H, E, L]
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # 2. 时延加权聚合 (分为训练与推断两种加速策略)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
    自相关机制多头包装投影层 (AutoCorrelation Multi-Head Layer)

    负责前向特征的多头线性投影、调用 AutoCorrelation 核心计算，并将多头特征投影还原。

    Args:
        correlation (nn.Module): 核心自相关计算模块 (AutoCorrelation)
        d_model (int): 隐藏特征通道维度
        n_heads (int): 多头注意力的头数
        d_keys (int, optional): Key 的头特征维度. 默认为 d_model // n_heads.
        d_values (int, optional): Value 的头特征维度. 默认为 d_model // n_heads.
    """

    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        张量流动追踪:
            queries: [B, L, d_model] -> query_projection -> [B, L, H, d_keys]
            keys: [B, S, d_model] -> key_projection -> [B, S, H, d_keys]
            values: [B, S, d_model] -> value_projection -> [B, S, H, d_values]
            inner_correlation 输出: [B, L, H, d_values]
            view 展平多头: [B, L, H * d_values]
            out_projection: [B, L, d_model]
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 多头投影
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 频域自相关与时延聚合计算
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        # 恢复维度并投影输出
        out = out.view(B, L, -1)
        return self.out_projection(out), attn