# -*- coding: utf-8 -*-
"""
模块说明: 多类型自注意力机制算法族 (layers/SelfAttention_Family.py)
-------------------------------------------------------------------------
该模块集中实现了面向时序特征建模的多种前沿自注意力计算层与封装结构：
1. `FullAttention`: 经典标准点积缩放全自注意力 (O(L^2) 复杂度)。
2. `ProbAttention`: Informer 提出的基于 KL 散度度量的概率稀疏自注意力 (ProbSparse Attention, O(L ln L) 复杂度)。
3. `DSAttention`: 去平稳自注意力 (De-stationary Attention)，引入自适应缩放因子 tau 与平移因子 delta 应对非平稳时序漂移。
4. `AttentionLayer`: 多头投影适配层，将高维特征线性映射为 Q/K/V 多头格式并包裹注意力算法。
5. `ReformerLayer`: 基于局部敏感哈希 (LSH) 的低内存消耗自注意力层。
6. `TwoStageAttentionLayer`: 跨时间与跨维度的两阶段时空解耦注意力层 (TSA Layer)。
"""

import math
from math import sqrt
import numpy as np
import torch
import torch.nn as nn

from util.masking import ProbMask, TriangularCausalMask

# 可选依赖项容错保护
try:
    from reformer_pytorch import LSHSelfAttention
except ImportError:
    LSHSelfAttention = None

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange, repeat = None, None


class DSAttention(nn.Module):
    """
    去平稳注意力模块 (De-stationary Attention)

    针对非平稳时间序列分布随时间漂移的问题，引入可学习的缩放因子 tau 与偏移因子 delta
    对计算得到的原始相关性分数矩阵进行调制重构。

    张量流动与变换追踪:
        queries: [B, L, H, E] (批大小, 查询序列长, 头数, 头维度)
        keys: [B, S, H, E] (批大小, 键序列长, 头数, 头维度)
        values: [B, S, H, D]
        scores: [B, H, L, S] (爱因斯坦求和 "blhe,bshe->bhls")
        调制后: scores = scores * tau + delta
        输出 V: [B, L, H, D]

    Args:
        mask_flag (bool, optional): 是否应用因果掩码. 默认为 True.
        factor (int, optional): 采样因子. 默认为 5.
        scale (float, optional): 点积缩放因子. 默认为 None (即 1/sqrt(E)).
        attention_dropout (float, optional): 注意力权重 Dropout. 默认为 0.1.
        output_attention (bool, optional): 是否返回权重矩阵. 默认为 False.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)    # [B, 1, 1, 1]
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 1. 爱因斯坦求和计算 Q 与 K 的批量内积得分: [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # 2. 因果掩码填充
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 3. Softmax 归一化注意力权重
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]
        # 4. 加权聚合 Value: [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    标准点积缩放全自注意力模块 (Full Scaled Dot-Product Attention)

    公式:
        scores = (Q @ K^T) / sqrt(d_k)
        attn = softmax(scores)
        out = attn @ V

    张量流动与变换追踪:
        queries: [B, L, H, E]
        keys: [B, S, H, E]
        values: [B, S, H, D]
        scores: [B, H, L, S]
        输出 V: [B, L, H, D]

    Args:
        mask_flag (bool, optional): 是否开启因果时序掩码. 默认为 True.
        factor (int, optional): 算法超参数. 默认为 5.
        scale (float, optional): 缩放因子. 默认为 None (即 1/sqrt(E)).
        attention_dropout (float, optional): 注意力 Dropout. 默认为 0.1.
        output_attention (bool, optional): 是否输出注意力矩阵. 默认为 False.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        # 1. 批量多头内积相关性得分: [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 2. 因果遮蔽
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 3. Softmax 概率归一化
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]
        # 4. 加权聚合 Value: [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    ProbSparse 概率稀疏自注意力模块 (Informer 创新算法)

    核心思想:
        通过随机采样子集估算每个 Query 的“活跃度” (KL 散度度量)，仅保留前 Top-u 个活跃 Query
        与全部 Key 计算注意力，非活跃 Query 直接取全局均值/累计上下文，将复杂度由 O(L^2) 降低至 O(L ln L)。

    Args:
        mask_flag (bool, optional): 是否应用因果掩码. 默认为 True.
        factor (int, optional): 采样敏感因子. 默认为 5.
        scale (float, optional): 缩放因子. 默认为 None.
        attention_dropout (float, optional): 注意力 Dropout. 默认为 0.1.
        output_attention (bool, optional): 是否输出完整权重图. 默认为 False.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        采样估算查询活跃度并挑选 Top-u 查询

        Args:
            Q (torch.Tensor): [B, H, L_Q, E]
            K (torch.Tensor): [B, H, L_K, E]
            sample_k (int): 随机抽样对比的 Key 数量
            n_top (int): 挑选出的主导活跃 Query 数量

        Returns:
            Q_K (torch.Tensor): 选出的 Top-u 查询与全量 Key 的得分矩阵 [B, H, n_top, L_K]
            M_top (torch.Tensor): Top-u 查询的索引张量 [B, H, n_top]
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 随机采样 sample_k 个 Key
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 基于最大值减均值度量稀疏活跃度 M: [B, H, L_Q]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # 挑选 Top-u 索引
        M_top = M.topk(n_top, sorted=False)[1]

        # 提取活跃 Query 并计算与全量 Key 的内积: [B, H, n_top, L_K]
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """为非活跃 Query 构建默认兜底上下文张量"""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """用活跃 Query 计算的注意力加权结果更新默认上下文"""
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        # 散弹更新活跃位置
        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # [B, H, L_Q, D]
        keys = keys.transpose(2, 1)        # [B, H, L_K, D]
        values = values.transpose(2, 1)    # [B, H, L_V, D]

        # 计算采样数量
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        # 稀疏采样与得分计算
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    通用多头注意力投影适配器 (Multi-Head Attention Layer)

    负责执行 4 个主要步骤:
    1. 通过线性层将 queries, keys, values 分别映射为多头子空间张量 [B, L, H, d_k]。
    2. 调用 inner_attention 执行具体的自注意力核心算法 (Full / Prob / DS Attention)。
    3. 合并多头通道为单一全局特征维度。
    4. 线性投影输出 [B, L, d_model]。

    Args:
        attention (nn.Module): 内部注意力机制实例
        d_model (int): 隐藏特征通道数
        n_heads (int): 多头注意力的头数
        d_keys (int, optional): Key 的头特征维度. 默认为 d_model // n_heads.
        d_values (int, optional): Value 的头特征维度. 默认为 d_model // n_heads.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        张量流动追踪:
            queries 输入: [B, L, d_model] -> query_projection -> [B, L, H, d_k]
            keys 输入: [B, S, d_model] -> key_projection -> [B, S, H, d_k]
            values 输入: [B, S, d_model] -> value_projection -> [B, S, H, d_v]
            inner_attention 输出: [B, L, H, d_v]
            view 展平多头: [B, L, H * d_v]
            out_projection: [B, L, d_model]
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 线性映射并拆分多头
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 计算核心注意力
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        # 合并多头并投影输出
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """局部敏感哈希注意力层 (Reformer Layer)"""

    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        if LSHSelfAttention is not None:
            self.attn = LSHSelfAttention(
                dim=d_model,
                heads=n_heads,
                bucket_size=bucket_size,
                n_hashes=n_hashes,
                causal=causal
            )
        else:
            self.attn = None

    def fit_length(self, queries):
        """填充序列长度使之被 bucket_size * 2 整除"""
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        B, N, C = queries.shape
        if self.attn is not None:
            queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    """
    两阶段时空解耦注意力层 (Two-Stage Attention, TSA)
    阶段 1: 沿时间维度独立执行自注意力 (Cross Time Stage)
    阶段 2: 沿特征维度通过共享路由器 (Router) 向量聚合信息 (Cross Dimension Stage)
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=configs.output_attention), d_model, n_heads
        )
        self.dim_sender = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=configs.output_attention), d_model, n_heads
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=configs.output_attention), d_model, n_heads
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if rearrange is None or repeat is None:
            return x

        batch = x.shape[0]
        # 1. 跨时间阶段
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None, tau=None, delta=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # 2. 跨空间/维度阶段 (基于可学习路由向量交互)
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
        return final_out