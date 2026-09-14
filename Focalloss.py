# -*- coding: utf-8 -*-
"""
模块说明: 损失函数模块 (Focalloss.py)
------------------------------------------------------------
该模块针对城市区域功能分类 (URFC) 任务中存在的极端类别不均衡问题（例如居住区、商业区
样本数量远多于绿地、仓储物流等长尾类别），实现了多种鲁棒的损失函数：
1. FocalLoss: 聚焦难分类样本的自适应加权损失函数，降低易分类负样本的梯度贡献。
2. Balanced_CE_loss: 样本级类别平衡交叉熵损失函数。
3. CombinedLoss: 融合标准交叉熵、类别平衡损失与 Focal Loss 的复合多重目标损失函数。
"""

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """
    多分类 Focal Loss 损失函数

    公式:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    通过引入调制因子 (1 - p_t)^gamma，使模型在训练时更加聚焦于难分类 (hard) 样本。

    Args:
        gamma (float, optional): 聚焦参数 (Focusing Parameter)，gamma >= 0.
            当 gamma=0 时退化为标准交叉熵; gamma 越大，对易分类样本的抑制效果越强. 默认为 2.
        alpha (float, optional): 类别权重平衡系数. 默认为 1.
        size_average (bool, optional): 是否对批次损失求均值 (True) 还是求和 (False). 默认为 True.
    """

    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 1e-6

    def forward(self, logits, labels):
        """
        前向计算 Focal Loss

        Args:
            logits (torch.Tensor): 未经 Softmax 的分类 Logits 输出
                形状为 [B, C] 或 [B, C, L]，其中 B=批大小, C=类别数 (如 9), L=序列长度 (可选)
            labels (torch.Tensor): 真实标签索引 (LongTensor)
                形状为 [B] 或 [B, L]，数值范围为 [0, C-1]

        Returns:
            torch.Tensor: 标量损失值
        """
        # 1. 扩展维度以便统一张量形状
        # logits: [B, C] -> [B, C, 1]
        # labels: [B] -> [B, 1]
        logits = logits[..., None]
        labels = labels[..., None]

        # 2. 针对多维情况调整形状
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()

        # 形状断言: 校验批次大小与序列长度匹配
        assert (logits.size(0) == labels.size(0)), f"批次大小不匹配: {logits.size(0)} vs {labels.size(0)}"
        assert (logits.size(2) == labels.size(1)), f"维度不匹配: {logits.size(2)} vs {labels.size(1)}"

        batch_size = logits.size(0)
        labels_length = logits.size(1)  # 对应分类类别数 C
        seq_length = logits.size(2)     # 序列长度 (默认 1)

        # 3. 将真实类别索引转换为 One-Hot 独热编码
        # new_label: [B, 1, 1]
        new_label = labels.unsqueeze(1)
        # label_onehot: [B, C, 1]
        label_onehot = torch.zeros([batch_size, labels_length, seq_length], device=logits.device).scatter_(1, new_label, 1)

        # 4. 计算 Log-Softmax 概率与真实标签处的对数似然
        # log_p: [B, C, 1]
        log_p = F.log_softmax(logits, dim=1)
        # pt: 仅保留真实类别所在位置的对数概率
        pt = label_onehot * log_p
        sub_pt = 1 - pt  # 调制因子底数 (1 - pt)

        # 5. 计算 Focal Loss 加权损失值
        # fl: [B, C, 1]
        fl = -self.alpha * (sub_pt ** self.gamma) * log_p

        # 6. 根据配置返回均值或总和
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class Balanced_CE_loss(torch.nn.Module):
    """
    类别平衡交叉熵损失函数 (Balanced Cross-Entropy Loss)

    通过统计样本内部的正负样本比例，动态生成平衡因子 beta，缓解极端的样本分布倾斜。
    """

    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        """
        前向计算平衡交叉熵

        Args:
            input (torch.Tensor): 预测概率或 Logits, 形状展平后为 [B, N]
            target (torch.Tensor): 目标独热标签, 形状展平后为 [B, N]

        Returns:
            torch.Tensor: 标量平衡交叉熵损失值
        """
        input = input.view(input.shape[0], -1).to(device)
        target = target.view(target.shape[0], -1).to(device)
        loss = 0.0

        for i in range(input.shape[0]):
            # 计算当前样本的正负比例加权因子 beta
            beta = 1 - torch.sum(target[i]) / target.shape[1]
            # 数值稳定截断: 避免 log(0) 导致产生 NaN / Inf
            x = torch.max(torch.log(input[i] + 1e-8), torch.tensor([-100.0]).to(device))
            y = torch.max(torch.log(1 - input[i] + 1e-8), torch.tensor([-100.0]).to(device))
            l = -(beta * target[i] * x + (1 - beta) * (1 - target[i]) * y)
            loss += torch.sum(l).to(device)

        return loss / input.shape[0]


class CombinedLoss(nn.Module):
    """
    多目标组合损失函数 (Combined Loss)

    将标准交叉熵损失 (CE)、类别平衡损失 (Class-Balanced CE) 与 Focal Loss 联合加权，
    综合兼顾整体收敛速度、长尾类别召回率以及难分类样本的挖掘。

    Args:
        alpha (float, optional): Focal Loss 类别平衡权重. 默认为 1.
        gamma (float, optional): Focal Loss 聚焦参数. 默认为 2.
        num_classes (int, optional): 分类类别总数. 默认为 9.
    """

    def __init__(self, alpha=1, gamma=2, num_classes=9):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

        # 1. 标准交叉熵损失
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # 2. 类别平衡交叉熵
        self.class_balance_loss = nn.CrossEntropyLoss(weight=None, reduction='mean')
        # 3. Focal Loss
        self.focal_loss = FocalLoss(gamma=self.gamma, alpha=self.alpha).to(device)

    def forward(self, inputs, targets):
        """
        前向计算复合损失均值

        Args:
            inputs (torch.Tensor): 模型预测 Logits，形状为 [B, num_classes] (如 [B, 9])
            targets (torch.Tensor): 真实目标类别索引，形状为 [B]

        Returns:
            torch.Tensor: 加权平均后的复合损失标量
        """
        # 分别计算三项子损失
        ce_loss = self.cross_entropy_loss(inputs, targets)
        cb_loss = self.class_balance_loss(inputs, targets)
        fl_loss = self.focal_loss(inputs, targets)

        # 联合等权平均
        return (ce_loss + cb_loss + fl_loss) / 3.0