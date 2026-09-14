# -*- coding: utf-8 -*-
"""
模块说明: 图像特征对时空到访分支的自注意力交互模块 (selfattentiontovis.py)
-------------------------------------------------------------------------
该模块实现了一个面向图像特征重排张量的点积缩放自注意力层 (Self-Attention Layer)，
在多模态融合网络 (MultiModalNet) 中用于对光学影像特征 (Image) 进行子空间注意力加权，
将图像的高维空间语义信息聚合后注入到访时空分支 (Image -> Visit)，实现反向模态交互。

计算过程:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
"""

from math import sqrt
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    图像到时空分支的点积缩放自注意力交互模块

    用于处理由 256 维图像特征重排得到的 [B, 64, 4] 张量 (64 个特征组，每组 4 维)。

    Args:
        dim_q (int, optional): 输入特征维度 / Query 原始维度. 默认为 4.
        dim_k (int, optional): 变换后的 Key 与 Query 隐藏维度. 默认为 64.
        dim_v (int, optional): 变换后的 Value 投影维度. 默认为 1.
    """

    def __init__(self, dim_q=4, dim_k=64, dim_v=1):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 线性投影层
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)  # Q 投影: 4 -> 64
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)  # K 投影: 4 -> 64
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)  # V 投影: 4 -> 1

        # 缩放因子: 1 / sqrt(d_k) = 1 / sqrt(64) = 0.125
        self._norm_fact = 1.0 / sqrt(dim_k)

    def forward(self, x):
        """
        前向传播与图像特征注意力汇聚

        Args:
            x (torch.Tensor): 输入图像重排特征张量
                形状为 [batch, n, dim_q]，在 URFC 中对应 [B, 64, 4]
                其中 B=批大小, 64=特征块序列长度, 4=子维度

        Returns:
            att (torch.Tensor): 经过注意力汇聚后的特征张量
                形状为 [batch, n, dim_v]，对应 [B, 64, 1]
        """
        # 1. 维度校验: x 形状为 [B, n, dim_q] = [B, 64, 4]
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q, f"输入特征维度 {dim_q} 与配置维度 {self.dim_q} 不一致"

        # 2. 线性变换得到 Query, Key, Value
        # x: [B, 64, 4] -> Linear -> q: [B, 64, 64]
        q = self.linear_q(x)
        # x: [B, 64, 4] -> Linear -> k: [B, 64, 64]
        k = self.linear_k(x)
        # x: [B, 64, 4] -> Linear -> v: [B, 64, 1]
        v = self.linear_v(x)

        # 3. 计算缩放点积相似度矩阵
        # q @ k^T: [B, 64, 64] @ [B, 64, 64] -> [B, 64, 64]
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact

        # 4. Softmax 归一化注意力权重
        # dist: [B, 64, 64] -> softmax -> [B, 64, 64]
        dist = torch.softmax(dist, dim=-1)

        # 5. 加权汇聚 Value
        # dist @ v: [B, 64, 64] @ [B, 64, 1] -> [B, 64, 1]
        att = torch.bmm(dist, v)

        return att
