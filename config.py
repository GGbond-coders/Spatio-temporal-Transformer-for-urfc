# -*- coding: utf-8 -*-
"""
模块说明: 全局超参数与运行环境配置模块 (config.py)
------------------------------------------------------------
该模块定义了城市区域功能分类 (URFC) 任务所需的全部超参数，包括：
1. 数据规格：遥感图像宽高与通道数、到访时空时序维度。
2. 训练超参：初始学习率、衰减率、权重衰减、批大小 (Batch Size)、迭代轮数 (Epochs)。
3. 文件与目录路径：训练集/测试集图像路径、.npy 时空文件路径。
4. 运行备份机制：每次初始化自动在 bak/ 目录下创建带时间戳的实验副本，并初始化检查点与日志目录。
"""

import os
import time
import warnings
from shutil import copyfile


class DefaultConfigs(object):
    """
    默认配置类: 管理模型训练、数据加载以及检查点路径等全局参数
    """
    # ------------------ 模型与任务配置 ------------------
    model_name = "Spatio-temporal Transformer"  # 模型标识名称
    num_classes = 9                             # 城市功能区分类类别数 (0~8共9类)

    # ------------------ 输入数据规格 ------------------
    # 光学遥感影像规格: [Channels=3, Height=100, Width=100]
    img_weight = 100                            # 原始图像宽度 (像素)
    img_height = 100                            # 原始图像高度 (像素)
    channels = 3                                # 图像通道数 (RGB)

    # 人群到访时空数据规格: [Days=7, Weeks=26, Hours=24]
    # 在数据集中展开后对应为 182 天 (7*26) × 24 小时
    vis_channels = 7                            # 周期维度: 每周 7 天 (周一至周日)
    vis_height = 26                             # 时序维度: 包含 26 周数据
    vis_weight = 24                             # 日时序维度: 每天 24 小时

    # ------------------ 优化器与训练超参数 ------------------
    lr = 0.0015                                 # 初始学习率 (Learning Rate)
    lr_decay = 0.5                              # 学习率衰减系数
    weight_decay = 0e-5                         # 权重衰减系数 (L2 正则化项)
    batch_size = 16                             # 训练批处理大小 (Batch Size)
    epochs = 30                                 # 最大训练迭代轮数

    # ------------------ 数据集存储路径 ------------------
    train_data = "./data/train/"                # 训练集光学影像存储目录 (.jpg)
    test_data = "./data/test/"                  # 测试集光学影像存储目录 (.jpg)
    train_vis = "./data/npy/train_visit"        # 训练集到访数据存储目录 (.npy)
    test_vis = "./data/npy/test_visit"          # 测试集到访数据存储目录 (.npy)
    load_model_path = None                      # 预训练权重恢复路径 (可选)

    def __init__(self):
        """
        初始化配置并创建实验备份目录:
        自动在 bak/ 目录下生成按时间戳命名的实验子目录，并将关键代码文件备份，
        防止实验过程中代码修改导致结果不可复现。
        """
        # 获取当前文件所在根目录
        root = os.path.dirname(os.path.abspath(__file__))
        bak_dir = os.path.join(root, "bak")
        os.makedirs(bak_dir, exist_ok=True)

        # 格式化当前时间戳: 月-日-时_分_秒
        time_now = time.strftime("%m-%d-%H_%M_%S", time.localtime())
        safe_model_name = self.model_name.replace(" ", "_")
        path = os.path.join(bak_dir, f"{time_now}_{safe_model_name}")
        os.makedirs(path, exist_ok=True)

        # 自动备份核心源码，保证可追溯性
        for fname in ["multimodal.py", "multimain.py", "config.py"]:
            src = os.path.join(root, fname)
            dst = os.path.join(path, fname)
            if os.path.exists(src):
                copyfile(src, dst)
        print('实验目录与源码已备份至: ' + path)

        # 动态绑定检查点、日志、提交结果输出路径
        self.weights = os.path.join(path, "checkpoints") + os.sep           # 权重存储根目录
        self.best_models = os.path.join(path, "checkpoints", "best_models") + os.sep  # 最佳模型目录
        self.logs = path + os.sep                                           # 训练日志目录
        self.debug_file = os.path.join(path, "tmp", "debug")                # 调试临时文件
        self.submit = os.path.join(path, "submit") + os.sep                 # 测试预测结果目录


def parse(self, kwargs):
    """
    通过字典或命令行参数动态更新配置对象中的属性值

    Args:
        kwargs (dict): 待更新的超参数键值对
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn(f"Warning: DefaultConfigs has no attribute '{k}'")
        setattr(self, k, v)

    print('当前生效的配置属性:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(f"  {k}: {getattr(self, k)}")


# 将 parse 方法绑定至配置类
DefaultConfigs.parse = parse
# 实例化全局单例配置对象
config = DefaultConfigs()
