# -*- coding: utf-8 -*-
"""
模块说明: Autoformer 时序渐进式分解编码器与解码器架构 (layers/Autoformer_EncDec.py)
---------------------------------------------------------------------------------
该模块实现了 Autoformer (NeurIPS 2021) 提出的渐进式时序分解架构 (Progressive Decomposition):
1. `moving_avg`: 移动平均池化层，平滑提取时序中的长期趋势项 (Trend Component)。
2. `series_decomp`: 时序分解模块，将输入时序解耦为周期季节项 (Seasonal) 与长期趋势项 (Trend)。
3. `my_Layernorm`: 专用于季节项的去均值层归一化。
4. `EncoderLayer` & `Encoder`: 编码器层在注意力与前馈计算中穿插时序分解，逐步过滤杂波。
5. `DecoderLayer` & `Decoder`: 解码器双分支交替累加趋势分量并建模周期分量，实现可解释的高精度时序建模。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    季节项专用去偏置层归一化 (Special LayerNorm for Seasonal Component)

    标准 LayerNorm 之后减去时间维度上的全局均值偏差，使周期特征围绕零点对称震荡。
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量 [B, L, C]

        Returns:
            torch.Tensor: 去偏置后的归一化张量 [B, L, C]
        """
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    时序移动平均池化层 (Moving Average Pooling)

    使用一维均值池化 (AvgPool1d) 平滑消除高频波动，凸显序列的长期趋势项。

    张量流动与变换追踪:
        输入 x: [B, L, C]
        首尾端点复制填充: front [B, pad, C], end [B, pad, C] -> 拼接为 [B, L + 2*pad, C]
        permute(0, 2, 1): [B, C, L + 2*pad]
        AvgPool1d(kernel_size, stride=1): [B, C, L]
        permute(0, 2, 1): [B, L, C] (恢复原始长度)

    Args:
        kernel_size (int): 移动平均滑动窗口大小 (如 25)
        stride (int): 步长 (通常为 1)
    """

    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 首尾两侧填充，避免序列因池化缩短
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)  # [B, L + 2*pad_len, C]

        # 均值滤波提取趋势项
        x = self.avg(x.permute(0, 2, 1))       # [B, C, L]
        x = x.permute(0, 2, 1)                 # [B, L, C]
        return x


class series_decomp(nn.Module):
    """
    时序双分量分解模块 (Series Decomposition Block)

    公式:
        Trend = MovingAvg(x)
        Seasonal = x - Trend

    Args:
        kernel_size (int): 移动平均平滑核尺寸
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 待分解的原始时序张量 [B, L, C]

        Returns:
            res (torch.Tensor): 季节/周期/高频残差分量 [B, L, C]
            moving_mean (torch.Tensor): 长期趋势/低频均值分量 [B, L, C]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer 渐进式分解编码器层 (Autoformer Encoder Layer)

    在自注意力机制与前馈网络的前后分别嵌入时序分解模块 (decomp1 与 decomp2)，
    确保编码器仅专注于对周期分量进行建模，消除趋势漂移的干扰。

    Args:
        attention (nn.Module): 自相关或自注意力层
        d_model (int): 隐藏特征通道数
        d_ff (int, optional): 前馈网络通道数. 默认为 4 * d_model.
        moving_avg (int, optional): 趋势平滑窗口尺寸. 默认为 25.
        dropout (float, optional): 失活概率. 默认为 0.1.
        activation (str, optional): 激活函数 ('relu' 或 'gelu'). 默认为 "relu".
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        张量流动追踪:
            x: [B, L, d_model]
            自注意力 + 残差: [B, L, d_model]
            第一阶段分解: x, _ = decomp1(x) -> 保留季节周期项 [B, L, d_model]
            前馈网络 FFN: [B, L, d_model] -> Conv1d -> [B, d_ff, L] -> Conv1d -> [B, L, d_model]
            第二阶段分解: res, _ = decomp2(x + y) -> [B, L, d_model]
        """
        # 1. 周期性自相关/自注意力计算
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        # 第一次分解过滤趋势项
        x, _ = self.decomp1(x)

        # 2. 前馈网络
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 第二次分解过滤前馈层产生的非平稳漂移
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """多层 Autoformer 编码器容器"""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer 渐进式分解解码器层 (Autoformer Decoder Layer)

    内部通过三阶段分解 (decomp1, decomp2, decomp3) 分别捕获自注意力、
    交叉注意力与前馈网络阶段产生的趋势项，并累加投影输出。

    Args:
        self_attention (nn.Module): 自回归掩码自注意力模块
        cross_attention (nn.Module): 跨层交叉注意力模块
        d_model (int): 隐藏状态维度
        c_out (int): 输出目标特征通道数
        d_ff (int, optional): 前馈网络通道数. 默认为 4 * d_model.
        moving_avg (int, optional): 趋势平滑窗口尺寸. 默认为 25.
        dropout (float, optional): 失活概率. 默认为 0.1.
        activation (str, optional): 激活函数 ('relu' 或 'gelu'). 默认为 "relu".
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        # 累积趋势项投影层
        self.projection = nn.Conv1d(
            in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
            padding_mode='circular', bias=False
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        张量流动追踪:
            x: [B, L_dec, d_model]
            cross: [B, L_enc, d_model]
            trend1, trend2, trend3: [B, L_dec, d_model]
            residual_trend: [B, L_dec, c_out]
            输出: x (季节分量) [B, L_dec, d_model], residual_trend (累计趋势分量) [B, L_dec, c_out]
        """
        # 1. 自注意力分支 + 第一阶段分解
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)

        # 2. 交叉注意力分支 + 第二阶段分解
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)

        # 3. 前馈神经网络 + 第三阶段分解
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # 4. 汇总三个阶段提取出的长期趋势分量并卷积投影
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer 多层解码器容器

    分别对季节周期项 (Seasonal) 与趋势项 (Trend) 进行累加，
    最终输出由两部分求和重构: Forecast = Seasonal + Trend
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        """
        Args:
            x (torch.Tensor): 季节初始化输入 [B, L_dec, d_model]
            cross (torch.Tensor): 编码器输出表征 [B, L_enc, d_model]
            x_mask (torch.Tensor, optional): 自注意力掩码
            cross_mask (torch.Tensor, optional): 交叉注意力掩码
            trend (torch.Tensor, optional): 趋势初始化输入 [B, L_dec, c_out]

        Returns:
            x (torch.Tensor): 解码完成的周期项
            trend (torch.Tensor): 最终累加聚合的趋势项
        """
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            if trend is not None:
                trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend