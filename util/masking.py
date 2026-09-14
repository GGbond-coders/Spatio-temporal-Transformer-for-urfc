# -*- coding: utf-8 -*-
"""
模块说明: 时序注意力掩码生成工具模块 (util/masking.py)
------------------------------------------------------------
该模块提供了在 Transformer 解码器与稀疏注意力机制中用于屏蔽未来时序信息或
非法注意力连接的掩码构造类：
1. TriangularCausalMask: 三角因果掩码 (上三角布尔矩阵)，用于自回归时间序列建模，
   防止当前时刻关注到未来时刻的信息。
2. ProbMask: 概率稀疏自注意力掩码，用于在 ProbSparse 注意力机制中根据挑选出的
   Top-u 活跃查询动态索引并对齐掩码。
"""

import torch


class TriangularCausalMask(object):
    """
    自回归时序三角因果掩码 (Triangular Causal Mask)

    生成形状为 [B, 1, L, L] 的上三角布尔张量，其中严格上三角 (对角线上方) 元素为 True，
    用于在 Attention 计算中将未来时间步的注意力权重置为 -inf。

    Args:
        B (int): 批次大小 (Batch Size)
        L (int): 时序序列长度 (Sequence Length)
        device (str, optional): 目标张量所在计算设备 ('cuda' 或 'cpu'). 默认为 "cpu".
    """

    def __init__(self, B, L, device="cpu"):
        # 掩码形状: [B, 1, L, L]
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # torch.triu 提取严格上三角矩阵 (diagonal=1 表示不含主对角线自身)
            # 严格上三角为 True (待屏蔽区域)，其余为 False (可见历史与当前)
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        获取构造好的布尔掩码张量

        Returns:
            torch.BoolTensor: 形状为 [B, 1, L, L]
        """
        return self._mask


class ProbMask(object):
    """
    ProbSparse 概率稀疏注意力专用掩码 (Prob Mask)

    在 Informer / ProbSparse 注意力中，仅对挑选出的部分查询 (Top Queries) 进行计算，
    本类负责根据查询的采样索引 index，将原始因果掩码重排并与采样注意力相似度矩阵 scores 对齐。

    Args:
        B (int): 批次大小 (Batch Size)
        H (int): 多头注意力的头数 (Number of Heads)
        L (int): 原始序列长度 (Sequence Length)
        index (torch.Tensor): 选出的活跃查询索引张量，形状为 [B, H, n_top]
        scores (torch.Tensor): 采样后计算得到的未归一化注意力得分矩阵，形状为 [B, H, n_top, S]
        device (str, optional): 计算设备. 默认为 "cpu".
    """

    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 1. 基础因果上三角掩码: [L, S]
        s = scores.shape[-1]
        _mask = torch.ones(L, s, dtype=torch.bool).to(device).triu(1)

        # 2. 广播扩展至多头批次维度: [B, H, L, S]
        _mask_ex = _mask[None, None, :].expand(B, H, L, s)

        # 3. 按照 Top-u 查询的索引张量 index 进行高维切片提取
        # indicator 形状: [B, H, n_top, S]
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ].to(device)

        # 4. 调整形状与 scores 矩阵严格一致: [B, H, n_top, S]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        获取对齐后的稀疏注意力掩码张量

        Returns:
            torch.BoolTensor: 形状与 scores 一致，为 [B, H, n_top, S]
        """
        return self._mask