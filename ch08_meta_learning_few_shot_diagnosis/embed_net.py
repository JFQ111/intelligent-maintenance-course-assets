"""embed_net.py — 共享 1D-CNN 嵌入网络，供 ProtoNet 和 MAML 复用。

网络结构与教材第八章 EmbedNet 一致：
    Conv1d(1, hidden, 8, stride=2) → BN → ReLU
    Conv1d(hidden, hidden, 5, stride=2) → BN → ReLU
    AdaptiveAvgPool1d(8) → Flatten → Linear(hidden*8, out_dim)

输入形状：(batch, 1, signal_length)
输出形状：(batch, out_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmbedNet(nn.Module):
    """轻量级 1D-CNN，用于从振动信号中提取特征向量。

    Parameters
    ----------
    in_channels : int
        输入通道数，振动信号通常为 1。
    hidden : int
        卷积层通道数。
    out_dim : int
        输出嵌入向量维度。
    """

    def __init__(self, in_channels: int = 1, hidden: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(hidden * 8, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EmbedNetWithHead(nn.Module):
    """EmbedNet + 线性分类头，供 MAML 内循环使用。

    MAML 的内循环需要一个完整的分类网络（特征提取 + 分类头），
    以便直接计算交叉熵损失。分类头的输出维度对应 n_way。

    Parameters
    ----------
    n_way : int
        分类头输出维度（episode 内类别数）。
    in_channels, hidden, out_dim : 同 EmbedNet。
    """

    def __init__(
        self,
        n_way: int,
        in_channels: int = 1,
        hidden: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = EmbedNet(in_channels=in_channels, hidden=hidden, out_dim=out_dim)
        self.classifier = nn.Linear(out_dim, n_way)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    def get_encoder(self) -> EmbedNet:
        """返回特征提取部分，便于 ProtoNet 复用已训练的编码器。"""
        return self.encoder
