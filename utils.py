# -*- coding: utf-8 -*-
"""
模块说明: 训练辅助与指标评估工具模块 (utils.py)
------------------------------------------------------------
该模块提供了深度学习训练过程中的常用辅助工具，包含：
1. 模型权重与检查点保存 (`save_checkpoint`)：支持自动持久化最新检查点并同步备份最佳准确率 (best_acc)、
   最佳损失 (best_loss) 及最佳 F1 分数 (best_f1) 的模型文件。
2. 运行指标统计器 (`AverageMeter`)：计算和追踪每个 Batch 的 Loss、Accuracy、F1 等指标的加权移动平均值。
3. 双向日志记录器 (`Logger`)：实现控制台标准输出 (stdout) 与日志文本文件 (.txt) 的同步写入。
4. 学习率提取与训练耗时格式化工具 (`get_learning_rate`, `time_to_str`)。
"""

import os
import sys
import shutil
import torch
from config import config


def save_checkpoint(model, state, is_best_acc, is_best_loss, is_best_f1, fold):
    """
    保存模型训练检查点并在达成历史最佳时同步归档

    Args:
        model (nn.Module): 当前训练的模型对象 (用于拓展兼容)
        state (dict): 需要持久化的训练状态字典，包含：
            - 'epoch': 当前迭代轮次
            - 'model_name': 模型名称
            - 'state_dict': 模型参数权重 OrderedDict
            - 'best_acc': 当前历史最高准确率
            - 'best_loss': 当前历史最低验证损失
            - 'optimizer': 优化器参数状态
            - 'fold': 当前折交叉验证编号
            - 'best_f1': 当前历史最高 Macro-F1 值
        is_best_acc (bool): 是否刷新了当前折的历史最高准确率
        is_best_loss (bool): 是否刷新了当前折的历史最低损失
        is_best_f1 (bool): 是否刷新了当前折的历史最高 Macro-F1
        fold (int): 交叉验证的折数标识 (从 0 开始)
    """
    # 确保当前折的检查点保存目录存在
    fold_dir = os.path.join(config.weights, config.model_name, str(fold))
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(config.best_models, exist_ok=True)

    # 保存常规 epoch 检查点文件
    filename = os.path.join(fold_dir, "checkpoint.pth.tar")
    torch.save(state, filename)

    # 若准确率达最佳，复制备份至 best_models 目录
    if is_best_acc:
        dst_acc = os.path.join(config.best_models, f"{config.model_name}_fold_{fold}_model_best_acc.pth.tar")
        shutil.copyfile(filename, dst_acc)

    # 若验证损失达最低，复制备份至 best_models 目录 (用于最终测试集加载推理)
    if is_best_loss:
        dst_loss = os.path.join(config.best_models, f"{config.model_name}_fold_{fold}_model_best_loss.pth.tar")
        shutil.copyfile(filename, dst_loss)

    # 若 Macro-F1 达最高，复制备份至 best_models 目录
    if is_best_f1:
        dst_f1 = os.path.join(config.best_models, f"{config.model_name}_fold_{fold}_model_best_f1.pth.tar")
        shutil.copyfile(filename, dst_f1)


class AverageMeter(object):
    """
    指标平均值统计器: 用于累加并计算训练与评估过程中的标量均值 (如 Loss, Accuracy, F1)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有计数器与累加值"""
        self.val = 0    # 当前批次 (batch) 的即时值
        self.avg = 0    # 当前所有批次的加权移动平均值
        self.sum = 0    # 累计总和
        self.count = 0  # 累计样本总数

    def update(self, val, n=1):
        """
        更新指标值

        Args:
            val (float): 当前批次的统计标量值
            n (int, optional): 当前批次的样本数 (权重). 默认为 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class Logger(object):
    """
    双向日志记录器: 支持同时输出内容至标准控制台 (sys.stdout) 与日志文件
    """

    def __init__(self):
        self.terminal = sys.stdout  # 绑定系统默认的标准输出设备
        self.file = None

    def open(self, file, mode='w'):
        """
        打开指定路径的日志文件

        Args:
            file (str): 日志文本文件路径
            mode (str, optional): 写入模式 ('w' 覆盖写, 'a' 追加写). 默认为 'w'.
        """
        # 自动创建父级目录
        os.makedirs(os.path.dirname(file), exist_ok=True)
        self.file = open(file, mode, encoding='utf-8')

    def write(self, message, is_terminal=1, is_file=1):
        """
        向终端或文件写入日志文本

        Args:
            message (str): 待写入的日志消息字符串
            is_terminal (int, optional): 是否输出至控制台 (1=是, 0=否). 默认为 1.
            is_file (int, optional): 是否输出至文本文件 (1=是, 0=否). 默认为 1.
        """
        # 过滤回车换行符 '\r' (防止进度刷新覆盖文件)
        if '\r' in message:
            is_file = 0

        # 写入控制台
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        # 写入文件
        if is_file == 1 and self.file is not None:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        """刷新缓冲区"""
        pass


def get_learning_rate(optimizer):
    """
    从优化器中提取当前第一参数组的学习率

    Args:
        optimizer (torch.optim.Optimizer): PyTorch 优化器实例

    Returns:
        float: 当前学习率数值
    """
    return optimizer.param_groups[0]['lr']


def time_to_str(t, mode='min'):
    """
    将秒数时间转换为格式化的人类易读字符串

    Args:
        t (float): 持续时长 (以秒为单位)
        mode (str, optional): 转换模式 ('min' 分钟模式, 'sec' 秒模式). 默认为 'min'.

    Returns:
        str: 格式化时间字符串 (如 ' 1 hr 15 min' 或 ' 45 min 20 sec')
    """
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError(f"不支持的格式化模式: {mode}")
