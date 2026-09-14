# -*- coding: utf-8 -*-
"""
模块说明: 多模态数据集加载与时空交互网络核心架构 (multimodal.py)
-------------------------------------------------------------------------
该模块是城市区域功能分类 (URFC) 任务的核心模型定义文件，包含：
1. 数据加载与增强 Dataset: `MultiModalDataset`
   - 遥感光学影像读取与图像增强 (翻转、旋转、错切、模糊等)。
   - 人群到访时空序列 `.npy` 数组读取与多维重排。
2. 学习率调度器: `CosineAnnealingLR` (余弦退火策略)。
3. 注意力组件族:
   - `Attention` & `MultiHeadedAttention`: 空间/时间分离多头注意力机制。
   - `ChannelAttentionModule` & `SpatialAttentionModule`: 通道与空间注意力模块 (CBAM)。
   - `SelfAttentionConv` & `ConvTransformerBLock`: 一维卷积时序自注意力模块。
4. 主多模态网络: `MultiModalNet`
   - 光学影像分支: 基于预训练 Vision Transformer (ViT-B/16) + 多尺度重构池化与反卷积层。
   - 到访时空分支: 基于 ConvTransformer 提取长序列周期模式。
   - 双向跨模态交互: Image -> Visit 与 Visit -> Image 双向自注意力对齐。
   - 深度辅助监督: 单模态辅助分类头与主融合分类头联合训练。
"""

import math
import os
import pathlib
import random
import sys
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models
from torchvision import transforms as T
from torchvision import transforms

from config import config
from selfattention import SelfAttention
from selfattentiontovis import SelfAttention as SelfAttentiontovis
from Transformer import Model as TransformerModel

# 可选第三方依赖安全导入保护
try:
    import pretrainedmodels
    from pretrainedmodels.models import *
except ImportError:
    pretrainedmodels = None

try:
    from imgaug import augmenters as iaa
except ImportError:
    iaa = None

# 设备自适应配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定全局随机种子以确保实验可复现
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
p = 0.1


# =========================================================================
# 1. 多模态数据集类 (MultiModalDataset)
# =========================================================================

class MultiModalDataset(Dataset):
    """
    多模态数据集加载器 (PyTorch Dataset)

    同时读取并配对地块的光学遥感图像 (.jpg) 与人群到访时空多维数组 (.npy)。

    Args:
        images_df (pd.DataFrame): 包含样本元数据的 DataFrame (需包含 'Id' 列，训练集需包含 'Target' 列)
        base_path (str or Path): 光学影像存储根目录
        vis_path (str or Path): 人群到访 .npy 文件存储根目录
        augument (bool, optional): 是否开启数据增强. 默认为 True.
        mode (str, optional): 数据模式 ('train', 'val', 'test'). 默认为 "train".
    """

    def __init__(self, images_df, base_path, vis_path, augument=True, mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        if not isinstance(vis_path, pathlib.Path):
            vis_path = pathlib.Path(vis_path)

        self.images_df = images_df.copy()
        self.augument = augument
        self.vis_path = vis_path
        # 将样本 Id 格式化为 6 位数字符串作为文件基础路径 (如 000001)
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / str(x).zfill(6))
        self.mode = mode

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.images_df)

    def __getitem__(self, index):
        """
        获取单一样本

        张量变换追踪:
            光学图像 X:
                原始读取: [100, 100, 3] (OpenCV BGR/RGB)
                数据增强后: [100, 100, 3]
                ToTensor后: [3, 100, 100] (浮点类型，范围归一化至 [0.0, 1.0])
            时空数据 visit:
                原始读取: [7, 26, 24] (周天数, 周数, 小时)
                transpose(1, 2, 0): [26, 24, 7] (周数, 小时, 周天数)
                ToTensor后: [7, 26, 24] 保持各时空维度排列

        Args:
            index (int): 样本索引

        Returns:
            X (torch.FloatTensor): 图像张量，形状为 [3, 100, 100]
            visit (torch.FloatTensor): 到访时空张量，形状为 [26, 7, 24]
            y (int or str): 分类标签 (训练模式) 或文件完整绝对路径 (测试模式)
        """
        # 1. 读取光学影像与到访数据
        X = self.read_images(index)
        # 原始维度转置为 (26, 24, 7)
        visit = self.read_npy(index).transpose(1, 2, 0)

        # 2. 标签提取
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())

        # 3. 图像增强 (仅在训练模式触发)
        if self.augument:
            X = self.augumentor(X)

        # 4. 转换为 PyTorch 张量格式
        X = T.Compose([T.ToPILImage(), T.ToTensor()])(X)
        visit = T.Compose([T.ToTensor()])(visit)

        return X.float(), visit.float(), y

    def read_images(self, index):
        """读取指定索引的 .jpg 光学遥感图像 [100, 100, 3]"""
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = cv2.imread(filename + '.jpg')
        return images

    def read_npy(self, index):
        """读取指定索引的 .npy 人群到访时空数组 [7, 26, 24]"""
        row = self.images_df.iloc[index]
        filename = os.path.basename(str(row.Id.absolute()))
        pth = os.path.join(self.vis_path.absolute(), filename + '.npy')
        visit = np.load(pth)
        return visit

    def augumentor(self, image):
        """
        光学影像数据增强流水线

        包括: 水平/垂直翻转、仿射旋转 (90/180/270度)、随机错切 (shear) 以及多种模糊平滑滤波。
        """
        if iaa is not None:
            augment_img = iaa.Sequential([
                iaa.Fliplr(0.5),  # 50% 概率水平翻转
                iaa.Flipud(0.5),  # 50% 概率垂直翻转
                iaa.SomeOf((0, 4), [
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Affine(shear=(-16, 16)),
                ]),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),      # 高斯模糊
                    iaa.AverageBlur(k=(2, 7)),       # 均值模糊
                    iaa.MedianBlur(k=(3, 11)),       # 中值滤波
                ]),
            ], random_order=True)
            return augment_img.augment_image(image)
        else:
            # 当未安装 imgaug 时的轻量回退方案
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
            return image


# =========================================================================
# 2. 学习率调度器与工具层 (LR Scheduler & FCViewer)
# =========================================================================

class _LRScheduler(object):
    """自定义学习率调度器基类"""

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' is not specified in param_groups[{i}]")
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnnealingLR(_LRScheduler):
    """
    余弦退火学习率调度器 (Cosine Annealing Learning Rate)
    学习率随周期按照余弦函数平滑衰减至最小值 eta_min
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class FCViewer(nn.Module):
    """全连接特征展平层: 将多维张量展平为二维 [Batch_Size, -1]"""

    def forward(self, x):
        return x.view(x.size(0), -1)


# =========================================================================
# 3. 空间/时间解耦注意力模块 (Decoupled Spatial-Temporal Attention)
# =========================================================================

class Attention(nn.Module):
    """
    点积缩放注意力机制 (Scaled Dot-Product Attention)

    公式:
        scores = (Q @ K^T) / sqrt(d_k)
        attn = softmax(scores)
        out = attn @ V
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        """
        Args:
            query (torch.Tensor): [..., N_q, D]
            key (torch.Tensor): [..., N_k, D]
            value (torch.Tensor): [..., N_v, D_v]

        Returns:
            p_val (torch.Tensor): 注意力聚合值
            p_attn (torch.Tensor): 注意力概率矩阵
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        p_attn = nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头时空解耦注意力层 (Multi-Headed Spatial/Temporal Attention)

    支持在空间维度 (mode='s') 或时序维度 (mode='t') 执行多头注意力计算。
    """

    def __init__(self, tokensize, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.head = head
        self.h, self.w = tokensize

        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)

    def forward(self, x, t):
        x = x.view(-1, 2432, 1, 4, 3)
        x = x.permute(0, 1, 2, 4, 3)
        x = x.contiguous().view(-1, 1, 3, 4)

        bt, _, n, c = x.size()
        b = bt // t
        c_h = c // self.head

        dev = x.device
        key = self.key_embedding(x).to(dev)
        query = self.query_embedding(x).to(dev)
        value = self.value_embedding(x).to(dev)

        if self.mode == 's':
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            att, _ = self.attention(query, key, value)
            att = att.permute(0, 1, 3, 2, 4).contiguous().view(bt, n, c)
        elif self.mode == 't':
            key = key.view(b, t, 2, self.h // 2, 2, self.w // 2, self.head, c_h)
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).view(b, 4, self.head, -1, c_h)
            query = query.view(b, t, 2, self.h // 2, 2, self.w // 2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).view(b, 4, self.head, -1, c_h)
            value = value.view(b, t, 2, self.h // 2, 2, self.w // 2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).view(b, 4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, 2, 2, self.head, t, self.h // 2, self.w // 2, c_h)
            att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, n, c)

        output = self.output_linear(att).to(dev)
        return output


class SpatialTemporalselfattention(nn.Module):
    """时空自注意力块: 结合多头注意力和平均池化降维"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2432, 64)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dev = x.device
        attention_block_s = MultiHeadedAttention(
            tokensize=[4, 3], d_model=4, head=4, mode='s'
        ).to(dev)
        output = attention_block_s(x, 2432).to(dev)
        out = output.view(-1, 2432, 4, 3)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        output = self.linear(out)
        output = self.dropout(output)
        return output


class Bottleneck(nn.Module):
    """DenseNet / ResNeXt 瓶颈残差单元"""

    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes + dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        out = F.relu(out)
        return out


def STAattention():
    return SpatialTemporalselfattention()


# =========================================================================
# 4. 通道注意力与空间注意力模块 (CBAM)
# =========================================================================

class ChannelAttentionModule(nn.Module):
    """通道注意力模块 (Channel Attention): 学习不同特征通道之间的相互依赖关系"""

    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    """空间注意力模块 (Spatial Attention): 聚焦关键空间地物特征区域"""

    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度分别求均值与最大值: [B, 1, H, W]
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接生成双通道空间描述子: [B, 2, H, W]
        out = torch.cat([avgout, maxout], dim=1)
        # 7x7 大卷积生成空间权重图: [B, 1, H, W]
        out = self.sigmoid(self.conv2d(out))
        return out


# =========================================================================
# 5. 一维卷积时序自注意力模块 (ConvTransformer Block)
# =========================================================================

class SelfAttentionConv(nn.Module):
    """
    一维卷积自注意力层 (1D Convolutional Self-Attention)

    采用 Conv1d 投影 Query 与 Key，在捕捉局部时序上下文的同时进行跨时序自注意力加权。

    Args:
        k (int): 时序序列长度 (URFC 到访天数 182)
        headers (int, optional): 多头注意力的头数. 默认为 8.
        kernel_size (int, optional): 局部卷积核尺寸. 默认为 1.
        mask_next (bool, optional): 是否对未来时刻进行掩码. 默认为 True.
        mask_diag (bool, optional): 是否掩码对角线自身. 默认为 False.
    """

    def __init__(self, k, headers=8, kernel_size=1, mask_next=True, mask_diag=False):
        super().__init__()
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers
        padding = (kernel_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)
        self.unifyheads = nn.Linear(k * h, k)

    def forward(self, x):
        """
        张量流动追踪:
            输入 x: [B, 26, 7, 24] 或展平为 [B, 182, 24]
            permute: [B, 24, 182] -> (b=B, t=24, k=182)
            输出: [B, 24, 182]
        """
        # 重塑并转置维度: 将 182 视为序列特征通道 k, 24 小时视为时序步长 t
        x = x.contiguous().view(-1, 182, 24)
        x = x.contiguous().permute(0, 2, 1)  # [B, 24, 182]

        b, t, k = x.size()
        assert self.k == k, f"时序维度 {k} 与配置维度 {self.k} 不匹配"
        h = self.headers

        # 转置至 Conv1d 所需格式 [B, k, t]
        x_trans = x.transpose(1, 2)
        x_padded = self.padding_opertor(x_trans)

        queries = self.toqueries(x_padded).view(b, k, h, t)
        keys = self.tokeys(x_padded).view(b, k, h, t)
        values = self.tovalues(x_trans).view(b, k, h, t)

        # 调整为标准注意力格式: [B, h, t, k]
        queries = queries.transpose(1, 2).transpose(2, 3)
        values = values.transpose(1, 2).transpose(2, 3)
        keys = keys.transpose(1, 2).transpose(2, 3)

        # 缩放因子
        queries = queries / (k ** 0.25)
        keys = keys / (k ** 0.25)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # 批量矩阵乘法计算注意力权重: [B*h, t, t]
        weights = torch.bmm(queries, keys.transpose(1, 2))

        # 可选因果掩码遮蔽未来信息
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0)
            else:
                indices = torch.triu_indices(t, t, offset=1)
            weights[:, indices[0], indices[1]] = float('-inf')

        weights = F.softmax(weights, dim=2)
        output = torch.bmm(weights, values)  # [B*h, t, k]
        output = output.view(b, h, t, k).transpose(1, 2).contiguous().view(b, t, k * h)

        return self.unifyheads(output)  # 恢复形状: [B, t=24, k=182]


class ConvTransformerBLock(nn.Module):
    """
    时序卷积 Transformer 块 (包含自注意力 + 残差连接 + LayerNorm + MLP)
    """

    def __init__(self, k, headers, kernel_size=24, mask_next=True, mask_diag=False, dropout_proba=0.2):
        super().__init__()
        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        self.dropout = nn.Dropout(p=dropout_proba)

    def forward(self, x, train=False):
        """
        张量流动追踪:
            x 输入: [B, 26, 7, 24] -> view [B, 182, 24] -> permute [B, 24, 182]
            经过 Attention + 残差 + LayerNorm + FFN + 残差 + LayerNorm
            输出: [B, 24, 182]
        """
        x = x.view(-1, 182, 24).permute(0, 2, 1)  # [B, 24, 182]

        # 1. 自注意力分支与第一个残差连接
        x = self.attention(x) + x
        if train:
            x = self.dropout(x)
        x = self.norm1(x)

        # 2. 前馈神经网络分支与第二个残差连接
        x = self.feedforward(x) + x
        x = self.norm2(x)
        x = self.dropout(x)

        return x  # [B, 24, 182]


def ConvTransformer():
    """实例化到访时空数据的 ConvTransformer 基础模块 (k=182, heads=1)"""
    return ConvTransformerBLock(182, 1)


# =========================================================================
# 6. 主多模态交互融合网络 (MultiModalNet)
# =========================================================================

class MultiModalNet(nn.Module):
    """
    城市区域功能多模态时空交互网络 (Spatio-temporal MultiModalNet)

    架构设计:
        1. 光学影像分支:
           - 输入 [B, 3, 100, 100] -> Resize [B, 3, 224, 224]
           - ViT-B/16 抽取高维全局语义表征 -> [B, 1000]
           - 特征网格化重构 [B, 1, 100, 10] -> MaxPool(2,2) -> [B, 1, 50, 5]
           - 连续转置卷积 ConvTranspose 多尺度重建得到 x1 [B, 364], x2 [B, 486]
           - 拼接复合特征 [1000 + 250 + 364 + 486 = 2100] -> Linear降维 -> [B, 256]
        2. 人群到访时空分支:
           - 输入 [B, 26, 7, 24] -> ConvTransformer 建模连续周期动态 -> [B, 24, 182]
           - 展平为 [B, 4368] -> Linear降维 -> [B, 64]
        3. 双向跨模态注意力交互:
           - Visit -> Image: [B, 24, 182] -> SelfAttention -> [B, 24, 1] -> 投影至 1 维拼入图像分支 -> [B, 257]
           - Image -> Visit: [B, 64, 4] -> SelfAttentiontovis -> [B, 64, 1] -> 投影至 1 维拼入到访分支 -> [B, 65]
        4. 双重融合与深度多任务分类:
           - 拼接跨模态特征 [257 + 65 = 322] -> 主分类器输出 [B, num_classes]
           - 辅助图像分类头: [B, 256] -> [B, num_classes]
           - 辅助到访分类头: [B, 64] -> [B, num_classes]

    Args:
        backbone1 (str, optional): 备用视觉主干名称. 默认为 "se_resnext101_32x4d".
        backbone2 (str, optional): 备用时空主干名称. 默认为 "dpn26".
        drop (float, optional): 全连接分类层的 Dropout 失活概率. 默认为 0.5.
        pretrained (bool, optional): 是否加载 ImageNet 预训练权重. 默认为 True.
    """

    def __init__(self, backbone1="se_resnext101_32x4d", backbone2="dpn26", drop=0.5, pretrained=True):
        super().__init__()

        # ------------------ 1. 跨模态注意力组件 ------------------
        self.vis_img_attention = SelfAttention()              # Visit -> Image 跨模态交互
        self.img_vis_attention = SelfAttentiontovis()          # Image -> Visit 跨模态交互
        self.visitConv_model = ConvTransformer()              # 到访分支时序自注意力主干

        # ------------------ 2. 图像分支 Vision Transformer ------------------
        # 兼容不同 torchvision 版本的预训练权重加载接口
        if hasattr(models, 'ViT_B_16_Weights'):
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.img_model16 = models.vit_b_16(weights=weights)
        else:
            self.img_model16 = models.vit_b_16(pretrained=pretrained)

        # 图像尺度缩放与特征多尺度重构层
        self.resize = transforms.Resize([224, 224], antialias=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )
        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)
        )

        # ------------------ 3. 特征投影与降维层 ------------------
        self.sclss = nn.Linear(2100, 256)      # 复合图像特征降维: 2100 -> 256
        self.toimgcls = nn.Linear(24, 1)       # 到访注意力特征压缩: 24 -> 1
        self.toviscls = nn.Linear(64, 1)       # 图像注意力特征压缩: 64 -> 1
        self.visscls = nn.Linear(4368, 64)     # 到访全序列特征降维: 4368 (24*182) -> 64

        # ------------------ 4. 分类器头 (主头 + 双辅助头) ------------------
        self.cls = nn.Linear(322, config.num_classes)      # 主分类头: 257(图) + 65(访) = 322 -> 9类
        self.dropout = nn.Dropout(drop)
        self.img_cls = nn.Linear(256, config.num_classes)  # 图像分支单模态辅助分类头
        self.vis_cls = nn.Linear(64, config.num_classes)   # 到访分支单模态辅助分类头

    def forward(self, x_img, x_vis):
        """
        多模态前向推理与特征交互

        Args:
            x_img (torch.Tensor): 光学遥感影像张量，形状为 [B, 3, 100, 100]
            x_vis (torch.Tensor): 人群到访时空张量，形状为 [B, 26, 7, 24]

        Returns:
            x_cat (torch.Tensor): 跨模态深度融合后的主分类 Logits，形状为 [B, num_classes] (如 [B, 9])
            x_imgsoft (torch.Tensor): 图像单模态辅助分类 Logits，形状为 [B, num_classes]
            x_vissoft (torch.Tensor): 到访单模态辅助分类 Logits，形状为 [B, num_classes]
        """
        # ==================== 1. 光学遥感影像分支 ====================
        # x_img: [B, 3, 100, 100] -> Resize -> [B, 3, 224, 224]
        x_img = self.resize(x_img)
        # ViT-B/16 抽取高层表征: [B, 3, 224, 224] -> [B, 1000]
        x_img = self.img_model16(x_img)

        b = x_img.size(0)
        # 规整为单通道特征图进行多尺度重构: [B, 1000] -> [B, 1, 100, 10]
        vit_output = x_img.view(b, 1, 100, 10)
        # 2x2 最大池化: [B, 1, 100, 10] -> [B, 1, 50, 5]
        pooled_output = self.pool(vit_output)

        # 第一次转置卷积: [B, 1, 50, 5] -> [B, 1, 52, 7] (52*7 = 364)
        x1 = self.conv_transpose1(pooled_output)
        # 第二次转置卷积: [B, 1, 52, 7] -> [B, 1, 54, 9] (54*9 = 486)
        x2 = self.conv_transpose2(x1)

        # 拼接多尺度重构表征: [B, 1000 + 250 + 364 + 486] = [B, 2100]
        x_img = torch.cat((
            vit_output.view(b, 1000),
            pooled_output.view(b, 250),
            x1.view(b, 364),
            x2.view(b, 486)
        ), dim=1)

        # 降维映射: [B, 2100] -> [B, 256]
        x_img = self.sclss(x_img)
        # 重构为交叉注意力格式: [B, 64, 4]
        x_img_tovis = x_img.view(b, 64, 4)
        x_img = self.dropout(x_img)
        # 图像辅助分类头输出: [B, 256] -> [B, num_classes]
        x_imgsoft = self.img_cls(x_img)

        # ==================== 2. 人群到访时空序列分支 ====================
        # x_vis: [B, 26, 7, 24] -> ConvTransformer -> [B, 24, 182]
        x_vis = self.visitConv_model(x_vis)

        # ==================== 3. 跨模态交互: 到访 -> 图像 (Visit -> Image) ====================
        # x_vis: [B, 24, 182] -> 自注意力加权 -> [B, 24, 1]
        x_vis_toimg = self.vis_img_attention(x_vis)
        # 展平并线性投影至单维增广标量: [B, 24] -> [B, 1]
        x_vis_toimg = self.toimgcls(x_vis_toimg.view(b, -1))
        # 注入图像分支: [B, 1] concat [B, 256] -> [B, 257]
        x_img = torch.cat((x_vis_toimg, x_img), dim=1)

        # 展平到访全时序表征并降维: [B, 24*182=4368] -> [B, 64]
        x_vis = self.visscls(x_vis.view(b, -1))
        x_vis = self.dropout(x_vis)
        # 到访辅助分类头输出: [B, 64] -> [B, num_classes]
        x_vissoft = self.vis_cls(x_vis)

        # ==================== 4. 跨模态交互: 图像 -> 到访 (Image -> Visit) ====================
        # x_img_tovis: [B, 64, 4] -> 自注意力加权 -> [B, 64, 1]
        x_img_tovis = self.img_vis_attention(x_img_tovis)
        # 展平并线性投影至单维增广标量: [B, 64] -> [B, 1]
        x_img_tovis = self.toviscls(x_img_tovis.view(b, -1))
        # 注入到访分支: [B, 1] concat [B, 64] -> [B, 65]
        x_vis = torch.cat((x_img_tovis, x_vis), dim=1)

        # ==================== 5. 双重主融合分类器 ====================
        # 拼接跨模态增强特征: [B, 257] concat [B, 65] -> [B, 322]
        x_cat = torch.cat((x_img, x_vis), dim=1)
        # 主分类头映射: [B, 322] -> [B, num_classes]
        x_cat = self.cls(x_cat)
        x_cat = self.dropout(x_cat)

        # 返回主分类结果与两个辅助分类结果供多任务联合优化
        return x_cat, x_imgsoft, x_vissoft