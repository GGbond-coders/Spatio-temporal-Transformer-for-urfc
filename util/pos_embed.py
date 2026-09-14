# -*- coding: utf-8 -*-
"""
模块说明: 2D/1D 正余弦位置编码与高分辨率插值工具 (util/pos_embed.py)
-------------------------------------------------------------------------
该模块提供了视觉与多维时空 Transformer 所需的位置编码生成与插值功能：
1. `get_2d_sincos_pos_embed`: 为 2D 图像网格生成二维绝对正余弦位置编码 (Sin-Cos Pos Embedding)。
2. `get_1d_sincos_pos_embed_from_grid`: 一维连续坐标的正余弦频率编码生成。
3. `interpolate_pos_embed`: 针对预训练模型与当前输入图像分辨率不一致的情况，
   通过双三次样条插值 (Bicubic Interpolation) 动态自适应调整位置编码尺寸。
"""

import numpy as np
import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    生成二维图像网格的正余弦位置编码 (2D Sine-Cosine Position Embedding)

    张量流动与变换追踪:
        网格尺寸: grid_size x grid_size (如 14x14)
        pos_embed 输出: [grid_size*grid_size, embed_dim] (若有 cls_token 则为 [1 + grid_size*grid_size, embed_dim])

    Args:
        embed_dim (int): 位置编码的特征维度 (必须为偶数)
        grid_size (int): 图像 Patch 划分的网格宽高大小 (如 224/16 = 14)
        cls_token (bool, optional): 是否在头部包含类别标记 [CLS] 的零向量占位. 默认为 False.

    Returns:
        np.ndarray: 生成的位置编码矩阵，形状为 [N_patches, embed_dim] 或 [1+N_patches, embed_dim]
    """
    # 1. 构造宽度与高度坐标序列: [0, 1, ..., grid_size - 1]
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # 2. 生成二维平面网格坐标点
    grid = np.meshgrid(grid_w, grid_h)  # 顺序为 (width, height)
    # grid: [2, grid_size, grid_size]
    grid = np.stack(grid, axis=0)

    # 3. 重塑形状为 [2, 1, grid_size, grid_size]
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # 4. 可选拼接 [CLS] 分类标记的位置编码占位符 (全零行)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从二维网格坐标数组中提取高低维度分解的正余弦位置编码

    Args:
        embed_dim (int): 总嵌入维度 (必须为偶数)
        grid (np.ndarray): 形状为 [2, 1, H, W] 的网格坐标数组

    Returns:
        np.ndarray: 形状为 [H*W, embed_dim] 的二维位置编码矩阵
    """
    assert embed_dim % 2 == 0, f"嵌入维度 embed_dim ({embed_dim}) 必须为偶数"

    # 将维度平分为两半: 前半部分编码高度 H，后半部分编码宽度 W
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]

    # 水平拼接得到完整特征: [H*W, D]
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    生成一维正余弦位置编码 (1D Sine-Cosine Position Embedding)

    计算公式:
        PE(pos, 2i)   = sin(pos / (10000^(2i / d)))
        PE(pos, 2i+1) = cos(pos / (10000^(2i / d)))

    Args:
        embed_dim (int): 输出特征维度 D (必须为偶数)
        pos (np.ndarray): 待编码的位置坐标数组，大小为 (M,)

    Returns:
        np.ndarray: 输出形状为 [M, D] 的一维位置编码矩阵
    """
    assert embed_dim % 2 == 0, f"嵌入维度 embed_dim ({embed_dim}) 必须为偶数"

    # 计算角频率因子: omega = 1 / (10000 ^ (2i / d))
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0 ** omega)  # 形状为 [D/2]

    # 外积计算 pos * omega: [M, 1] * [1, D/2] -> [M, D/2]
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    # 分别计算正弦与余弦响应
    emb_sin = np.sin(out)  # [M, D/2]
    emb_cos = np.cos(out)  # [M, D/2]

    # 沿特征维度拼接: [M, D]
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    """
    针对不同分辨率输入动态双三次样条插值调整位置编码 (Interpolate Position Embeddings)

    当预训练模型（如 ImageNet 224x224 分辨率）迁移至更高分辨率（如 384x384 或 448x448）时，
    Patch 数量发生变化，该函数可对预训练的位置编码进行 2D 双三次平滑插值，保证特征对齐。

    Args:
        model (nn.Module): 当前构建的 ViT 目标网络实例
        checkpoint_model (dict): 从权重文件加载的状态字典 (state_dict)
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # 计算检查点原始网格边长与当前模型目标网格边长
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        # 仅在分辨率网格尺寸不一致时执行插值
        if orig_size != new_size:
            print(f"位置编码插值: 从 {orig_size}x{orig_size} 网格调整至 {new_size}x{new_size} 网格")
            # 保留 [CLS] 或蒸馏额外标记
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

            # 重塑为 2D 图像特征图格式以便双三次插值: [1, C, H_orig, W_orig]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            # 恢复为展平序列格式: [1, new_patches, C]
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

            # 重新拼合额外标记与插值后的位置编码
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
