# -*- coding: utf-8 -*-
"""
模块说明: 多模态模型主训练、验证与推断执行流水线 (multimain.py)
-------------------------------------------------------------------------
该模块是项目的总入口脚本，负责串联完整的数据流动与模型生命周期管理：
1. 环境配置与随机种子固化 (支持多卡/单卡/CPU 自由配置)。
2. 数据划分与加载：将 train.csv 划分为训练集与验证集 (9:1 划分)，构建多进程 DataLoader。
3. 训练循环 (train):
   - 支持图像与时空多模态联合输入。
   - 深度多任务监督: 主交叉熵损失 + 0.25*(图像辅助损失 + 到访辅助损失)。
   - 实时计算与输出 Accuracy、Macro-F1 分数以及 9x9 混淆矩阵。
4. 验证循环 (evaluate):
   - 在验证集上无梯度推断，监测泛化指标并动态判定最佳模型 (best_acc / best_loss / best_f1)。
5. 预测与交付 (test):
   - 载入验证集表现最佳的检查点，对测试集进行推理预测，生成符合比赛或业务规范的 CSV 结果文件。

运行命令:
    python multimain.py
"""

from __future__ import print_function
from datetime import datetime
import json
import os
import random
import sys
import time
from timeit import default_timer as timer
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from config import config
from Focalloss import Balanced_CE_loss, CombinedLoss, FocalLoss
from multimodal import CosineAnnealingLR, MultiModalDataset, MultiModalNet
from utils import AverageMeter, Logger, get_learning_rate, save_checkpoint, time_to_str

# ------------------ 1. 硬件设备与运行配置 ------------------
# 优先遵循系统已有的 CUDA_VISIBLE_DEVICES，默认回退至 GPU 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定全局随机种子以确保每次实验数据划分与初始化的确定性
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# 初始化双向训练日志器
log = Logger()
log.open(f"{config.logs}/{config.model_name}_log_train.txt", mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')


# =========================================================================
# 2. 训练函数 (train)
# =========================================================================

def train(train_loader, model, criterion, optimizer, epoch, valid_metrics, best_results, start):
    """
    单轮训练主函数 (包含多任务损失计算与反向传播)

    张量流动与变换追踪:
        images: [B, 3, 100, 100] (光学遥感影像)
        visit: [B, 26, 7, 24] (时空到访序列)
        target: [B] (类别整数标签 0~8)
        model(images, visit) 输出:
            output: [B, 9] (主融合分支 Logits)
            output1: [B, 9] (图像单模态辅助 Logits)
            output2: [B, 9] (到访单模态辅助 Logits)
        联合损失计算:
            loss = L(output, target) + 0.25 * (L(output1, target) + L(output2, target))

    Args:
        train_loader (DataLoader): 训练数据加载器
        model (nn.Module): 多模态网络实例
        criterion (nn.Module): 损失函数 (如 CrossEntropyLoss)
        optimizer (Optimizer): 优化器 (如 SGD)
        epoch (int): 当前迭代轮次 (从 0 开始)
        valid_metrics (list): 上一轮验证集指标 [acc, loss, f1]
        best_results (list): 历史最优指标 [best_acc, best_loss, best_f1]
        start (float): 训练启动的时间戳 (秒)

    Returns:
        list: 当前轮次训练集综合指标 [acc.avg, losses.avg, f1.avg, b (混淆矩阵)]
    """
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    # 初始化 9x9 分类混淆矩阵
    b = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

    model.train()

    for i, (images, visit, target) in enumerate(train_loader):
        # 1. 数据迁移至目标计算设备 (GPU / CPU)
        images = images.to(device)  # [B, 3, 100, 100]
        visit = visit.to(device)    # [B, 26, 7, 24]
        target = torch.from_numpy(np.array(target)).long().to(device)  # [B]

        # 2. 单次前向推理同时获取主融合预测与双辅助预测 (避免多次前向开销)
        output, output1, output2 = model(images, visit)

        # 3. 联合多任务深度监督损失 (辅助分支权重 a=0.25)
        a = 0.25
        loss = criterion(output, target) + a * (criterion(output1, target) + criterion(output2, target))

        # 4. 评估指标计算: Softmax 概率 -> Argmax 类别预测
        # pred: [B] (整数预测类别 0~8)
        pred = np.argmax(F.softmax(output, dim=1).detach().cpu().numpy(), axis=1)
        target_np = target.cpu().numpy()

        # 计算批次 Macro-F1 与准确率 Accuracy
        f1_batch = f1_score(target_np, pred, average='macro')
        acc_score = accuracy_score(target_np, pred)
        # 累加混淆矩阵
        com_c = confusion_matrix(target_np, pred, labels=list(range(config.num_classes)))
        b = b + com_c

        # 5. 更新统计量
        losses.update(loss.item(), images.size(0))
        f1.update(f1_batch, images.size(0))
        acc.update(acc_score, images.size(0))

        # 6. 反向传播与参数优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. 实时格式化输出训练进度
        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (
            "train", i / len(train_loader) + epoch, epoch,
            acc.avg, losses.avg, f1.avg,
            valid_metrics[0], valid_metrics[1], valid_metrics[2],
            str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
            time_to_str((timer() - start), 'min')
        )
        print(message, end='', flush=True)

    log.write("\n")
    print("\n训练集混淆矩阵:\n", b)
    return [acc.avg, losses.avg, f1.avg, b]


# =========================================================================
# 3. 验证评估函数 (evaluate)
# =========================================================================

def evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start):
    """
    模型验证集评估函数 (禁用梯度计算)

    Args:
        val_loader (DataLoader): 验证集数据加载器
        model (nn.Module): 多模态网络模型
        criterion (nn.Module): 损失函数
        epoch (int): 当前迭代轮次
        train_metrics (list): 当前轮训练集指标
        best_results (list): 历史最优指标
        start (float): 训练起始时间戳

    Returns:
        list: 验证集指标 [acc.avg, losses.avg, f1.avg]
    """
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    b = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, visit, target) in enumerate(val_loader):
            images_var = images.to(device)
            visit = visit.to(device)
            target = torch.from_numpy(np.array(target)).long().to(device)

            # 前向计算
            output, output1, output2 = model(images_var, visit)
            a = 0.25
            loss = criterion(output, target) + a * (criterion(output1, target) + criterion(output2, target))

            losses.update(loss.item(), images_var.size(0))

            # 类别预测与指标计算
            pred = np.argmax(F.softmax(output, dim=1).detach().cpu().numpy(), axis=1)
            target_np = target.cpu().numpy()
            f1_batch = f1_score(target_np, pred, average='macro')
            acc_score = accuracy_score(target_np, pred)
            com_c = confusion_matrix(target_np, pred, labels=list(range(config.num_classes)))
            b = b + com_c

            f1.update(f1_batch, images.size(0))
            acc.update(acc_score, images.size(0))

            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (
                "val", i / len(val_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                train_metrics[0], train_metrics[1], train_metrics[2],
                str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
                time_to_str((timer() - start), 'min')
            )
            print(message, end='', flush=True)

        log.write("\n")
        print("\n验证集混淆矩阵:\n", b)

    return [acc.avg, losses.avg, f1.avg]


# =========================================================================
# 4. 测试集推断与提交文件生成 (test)
# =========================================================================

def test(test_loader, model, folds):
    """
    测试集无标签样本推断并导出 CSV 结果文件

    Args:
        test_loader (DataLoader): 测试集 DataLoader (batch_size=1)
        model (nn.Module): 载入了最佳权重的网络模型
        folds (int): 当前折编号
    """
    sample_submission_df = pd.read_csv("./test.csv")
    filenames, labels = [], []

    model.to(device)
    model.eval()

    print("\n正在对测试集样本进行前向推断...")
    for i, (input_img, visit, filepath) in tqdm(enumerate(test_loader), total=len(test_loader)):
        filepath = [os.path.basename(x) for x in filepath]

        with torch.no_grad():
            image_var = input_img.to(device)
            visit = visit.to(device)

            # 获取主融合分类输出: y_pred 形状为 [1, 9]
            outputs = model(image_var, visit)
            y_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # Softmax 归一化生成 9 类的置信度概率分布
            prob = F.softmax(y_pred, dim=1).detach().cpu().numpy()
            labels.append(prob)
            filenames.append(filepath)

    # 汇总全部样本预测概率并求取最大置信度索引
    predictions = np.concatenate(labels, axis=0)  # [N_test, 9]
    submissions = np.argmax(predictions, axis=1)  # [N_test]

    sample_submission_df['Predicted'] = submissions
    submit_path = os.path.join(config.submit, f"{config.model_name}_bestloss_submission.csv")
    sample_submission_df.to_csv(submit_path, index=None)
    print("测试集预测完成! 结果文件已成功保存至:", submit_path)


# =========================================================================
# 5. 主训练管线入口 (main)
# =========================================================================

def main():
    """
    主训练入口函数:
    完成目录初始化、模型构建、优化器配置、数据集划分与训练/验证/测试循环
    """
    fold = 0

    # 1. 确保全部输出目录结构存在
    os.makedirs(config.submit, exist_ok=True)
    os.makedirs(os.path.join(config.weights, config.model_name, str(fold)), exist_ok=True)
    os.makedirs(config.best_models, exist_ok=True)
    os.makedirs(config.logs, exist_ok=True)

    # 2. 构建核心多模态网络模型
    model = MultiModalNet("se_resnext101_32x4d", "dpn26", drop=0.5, pretrained=True)

    # 3. 配置带动量的 SGD 优化器与权重衰减 (L2 正则化)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)

    # 4. 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_results = [0, np.inf, 0]  # 分别记录 [最高Acc, 最低Loss, 最高F1]
    val_metrics = [0, np.inf, 0]

    # 可选断点续训 (Resume Checkpoint)
    resume = False
    if resume and config.load_model_path:
        checkpoint = torch.load(config.load_model_path)
        best_results[0] = checkpoint.get('best_acc', 0)
        best_results[1] = checkpoint.get('best_loss', np.inf)
        best_results[2] = checkpoint.get('best_f1', 0)
        start_epoch = checkpoint.get('epoch', 0)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"成功恢复检查点: 从第 {start_epoch} 轮继续训练")

    # 多卡数据并行模式适配
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 5. 读取表格标注并进行 9:1 训练/验证集切分
    all_files = pd.read_csv("./train.csv")
    test_files = pd.read_csv("./test.csv")
    train_data_list, val_data_list = train_test_split(all_files, test_size=0.1, random_state=2050)

    # 6. 构造 PyTorch DataLoader
    # 训练集: 开启数据增强与随机打乱
    train_gen = MultiModalDataset(train_data_list, config.train_data, config.train_vis, augument=True, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    # 验证集: 关闭数据增强，顺序读取
    val_gen = MultiModalDataset(val_data_list, config.train_data, config.train_vis, augument=False, mode="train")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    # 测试集: 单样本读取推断
    test_gen = MultiModalDataset(test_files, config.test_data, config.test_vis, augument=False, mode="test")
    test_loader = DataLoader(test_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    # 7. 学习率动态调整调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    start = timer()

    # 8. 周期训练与评估主循环
    reslist = []
    for epoch in range(start_epoch, config.epochs):
        # 执行一轮完整训练
        train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)

        # 执行一轮验证集评估
        val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)

        # 更新学习率调度器 (基于验证集损失)
        scheduler.step(val_metrics[1])

        # 检查是否刷新历史最优指标
        is_best_acc = val_metrics[0] > best_results[0]
        best_results[0] = max(val_metrics[0], best_results[0])

        is_best_loss = val_metrics[1] < best_results[1]
        best_results[1] = min(val_metrics[1], best_results[1])

        is_best_f1 = val_metrics[2] > best_results[2]
        best_results[2] = max(val_metrics[2], best_results[2])

        # 持久化保存当前检查点并同步备份历史最佳权重
        save_checkpoint(
            model,
            {
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "best_acc": best_results[0],
                "best_loss": best_results[1],
                "optimizer": optimizer.state_dict(),
                "fold": fold,
                "best_f1": best_results[2],
            },
            is_best_acc, is_best_loss, is_best_f1, fold
        )

        # 日志记录轮次最优总结
        print('\r', end='', flush=True)
        log.write('%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s\n' % (
            "best", epoch, epoch,
            train_metrics[0], train_metrics[1], train_metrics[2],
            val_metrics[0], val_metrics[1], val_metrics[2],
            str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
            time_to_str((timer() - start), 'min')
        ))
        reslist.append({"epoch": epoch, "acc": str(best_results[0])[:5], "f1": str(best_results[2])[:5]})

    # 9. 加载验证集 Loss 最优的模型权重，对测试集进行推理输出
    best_loss_model_path = os.path.join(config.best_models, f"{config.model_name}_fold_{fold}_model_best_loss.pth.tar")
    print(f"\n正在加载历史最低损失权重进行推断: {best_loss_model_path}")
    best_model = torch.load(best_loss_model_path, weights_only=False)
    model.load_state_dict(best_model["state_dict"])

    # 执行测试预测
    test(test_loader, model, fold)


if __name__ == "__main__":
    main()
