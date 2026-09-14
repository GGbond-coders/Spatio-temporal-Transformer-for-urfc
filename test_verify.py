# -*- coding: utf-8 -*-
"""
模块说明: 多模态模型前向传播与反向传播连通性验证脚本 (test_verify.py)
-------------------------------------------------------------------------
该脚本用于在无须准备庞大真实数据集的前提下，快速验证本地 Python / PyTorch 环境配置：
1. 测试 MultiModalNet 模型的实例化 (关闭 pretrained 以便快速离线测试)。
2. 测试批次训练前向传播 (batch_size=2) 与多分支输出张量维度校验。
3. 测试测试集单样本推断模式 (batch_size=1) 与动态维度自适应。
4. 测试复合加权损失函数计算与反向传播梯度流 (loss.backward())。
5. 测试检查点保存 (save_checkpoint) 与参数反序列化加载 (load_state_dict)。

运行命令:
    python test_verify.py
"""

import os
import sys

# 动态绑定项目根目录至模块搜索路径
repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

import torch
from multimodal import MultiModalNet
from utils import save_checkpoint
from config import config


def test_model_forward():
    """
    执行完整的模型前向、反向与检查点单元测试
    """
    # ------------------ 1. 模型实例化测试 ------------------
    print("1. 正在测试 MultiModalNet 模型初始化 (pretrained=False)...")
    # 关闭预训练权重下载，仅校验网络结构构建与层参数连接
    model = MultiModalNet(pretrained=False)
    model.eval()

    # ------------------ 2. 训练批次 (batch_size = 2) 前向测试 ------------------
    print("2. 正在测试训练批次前向计算 (batch_size = 2)...")
    # 构造模拟光学影像输入: [B=2, C=3, H=100, W=100]
    x_img = torch.randn(2, 3, 100, 100)
    # 构造模拟时空到访输入: [B=2, T_weeks=26, D_days=7, H_hours=24]
    x_vis = torch.randn(2, 26, 7, 24)

    # 前向推理返回三个分支输出: 主融合输出、图像辅助输出、到访辅助输出
    out, out_img, out_vis = model(x_img, x_vis)
    print(f"   输出张量形状: out={out.shape}, out_img={out_img.shape}, out_vis={out_vis.shape}")

    # 验证分类维度是否严格符合类别数 (9类)
    assert out.shape == (2, config.num_classes), f"预期输出形状为 (2, {config.num_classes})，实际为 {out.shape}"
    assert out_img.shape == (2, config.num_classes), "图像辅助分类器输出维度异常"
    assert out_vis.shape == (2, config.num_classes), "到访辅助分类器输出维度异常"

    # ------------------ 3. 推理批次 (batch_size = 1) 单样本测试 ------------------
    print("3. 正在测试测试集单样本推断模式 (batch_size = 1)...")
    x_img_1 = torch.randn(1, 3, 100, 100)
    x_vis_1 = torch.randn(1, 26, 7, 24)
    out_1, _, _ = model(x_img_1, x_vis_1)
    print(f"   单样本推断输出形状: {out_1.shape}")
    assert out_1.shape == (1, config.num_classes), "单样本推断输出维度异常"

    # ------------------ 4. 损失计算与反向传播测试 ------------------
    print("4. 正在测试复合损失计算与反向梯度传播...")
    model.train()
    target = torch.tensor([0, 1], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    out, out_img, out_vis = model(x_img, x_vis)
    # 联合主损失与辅助分类器深度监督
    loss = criterion(out, target) + 0.25 * (criterion(out_img, target) + criterion(out_vis, target))
    loss.backward()
    print(f"   损失数值计算正常: loss = {loss.item():.4f}，反向传播成功!")

    # ------------------ 5. 模型检查点保存与反序列化测试 ------------------
    print("5. 正在测试检查点保存与参数重载...")
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "best_acc": 0.85,
        "best_loss": 0.5,
        "best_f1": 0.82
    }
    save_checkpoint(model, state, is_best_acc=True, is_best_loss=True, is_best_f1=True, fold=0)
    best_pth = os.path.join(config.best_models, f"{config.model_name}_fold_0_model_best_loss.pth.tar")
    assert os.path.exists(best_pth), f"检查点文件未成功生成: {best_pth}"

    # 加载权重
    loaded = torch.load(best_pth, weights_only=False)
    assert "state_dict" in loaded, "检查点字典中缺失 'state_dict' 键"
    model.load_state_dict(loaded["state_dict"])
    print("   检查点序列化与模型加载验证成功!")

    print("\n恭喜! 所有测试项全部通过 (ALL CHECKS PASSED)!")


if __name__ == "__main__":
    test_model_forward()
