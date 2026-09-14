# -*- coding: utf-8 -*-
"""
模块说明: 时空到访特征自注意力交互模块 (selfattention.py)
------------------------------------------------------------
该模块实现了一个基于标准点积缩放机制的自注意力层 (Self-Attention Layer)，
在多模态融合网络 (MultiModalNet) 中用于对人群到访时空特征 (Visit) 进行内部相关性建模
并压缩映射，为后续注入图像分支 (Visit -> Image) 提供交互特征权重。

计算过程:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
"""

from math import sqrt
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    点积缩放自注意力模块 (Self-Attention Module)

    用于时序维度为 182 (26周*7天)、特征维度为 24 (小时) 的时空特征矩阵建模。

    Args:
        dim_q (int, optional): 输入特征维度 / Query 原始维度. 默认为 182 (天数).
        dim_k (int, optional): 变换后的 Key 与 Query 隐藏维度. 默认为 24.
        dim_v (int, optional): 变换后的 Value 投影维度. 默认为 1.
    """

    def __init__(self, dim_q=182, dim_k=24, dim_v=1):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 线性投影矩阵: 将输入维度 dim_q 映射至特征子空间
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)  # Q 投影层: 182 -> 24
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)  # K 投影层: 182 -> 24
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)  # V 投影层: 182 -> 1

        # 缩放因子: 1 / sqrt(d_k)，用于避免内积过大导致 Softmax 梯度饱和
        self._norm_fact = 1.0 / sqrt(dim_k)

    def forward(self, x):
        """
        前向传播与注意力计算

        Args:
            x (torch.Tensor): 输入时空特征张量
                形状为 [batch, n, dim_q]，在 URFC 中对应 [B, 24, 182]
                其中 B=批大小, 24=小时特征通道, 182=连续天数序列

        Returns:
            att (torch.Tensor): 经过注意力加权聚合后的特征张量
                形状为 [batch, n, dim_v]，对应 [B, 24, 1]
        """
        # 1. 维度校验: x 形状为 [B, n, dim_q] = [B, 24, 182]
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q, f"输入特征维度 {dim_q} 与配置维度 {self.dim_q} 不一致"

        # 2. 线性投影计算 Query, Key, Value
        # x: [B, 24, 182] -> Linear -> q: [B, 24, 24]
        q = self.linear_q(x)
        # x: [B, 24, 182] -> Linear -> k: [B, 24, 24]
        k = self.linear_k(x)
        # x: [B, 24, 182] -> Linear -> v: [B, 24, 1]
        v = self.linear_v(x)

        # 3. 计算缩放点积相似度矩阵 (Scaled Dot-Product)
        # q @ k^T: [B, 24, 24] @ [B, 24, 24] -> [B, 24, 24]
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact

        # 4. 在最后一个维度做 Softmax 归一化，生成注意力概率分布矩阵
        # dist: [B, 24, 24] -> softmax -> [B, 24, 24] (每行和为 1)
        dist = torch.softmax(dist, dim=-1)

        # 5. 注意力权重矩阵加权求和 Value
        # dist @ v: [B, 24, 24] @ [B, 24, 1] -> [B, 24, 1]
        att = torch.bmm(dist, v)

        return att
