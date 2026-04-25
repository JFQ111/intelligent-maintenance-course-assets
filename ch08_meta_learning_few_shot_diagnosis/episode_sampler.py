"""episode_sampler.py — 元学习 episode 采样与标签重映射核心模块。

本模块提供两个类：
- EpisodeSampler：从 npz 数据集中随机采样 N-way K-shot episode，
  并自动完成标签重映射，供 ProtoNet 和 MAML 直接使用。
- FewShotDataset：对 numpy 数组的轻量封装，提供按类别索引的接口。
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class Episode(NamedTuple):
    """单个 episode 的数据容器。

    Attributes
    ----------
    support_x : np.ndarray, shape (N*K, 1, L)
        支持集信号，已添加 channel 维度，标签已重映射。
    support_y : np.ndarray, shape (N*K,)
        支持集标签（0 到 N-1，episode 内连续整数）。
    query_x : np.ndarray, shape (N*Q, 1, L)
        查询集信号。
    query_y : np.ndarray, shape (N*Q,)
        查询集标签（与 support_y 同一映射）。
    label_map : dict[int, int]
        原始全局标签 → episode 内临时标签的映射关系，便于溯源。
    """

    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    label_map: dict[int, int]


class FewShotDataset:
    """按类别索引的轻量数据集封装。

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, signal_length)
        所有样本的一维信号（float32）。
    labels : np.ndarray, shape (n_samples,)
        全局类别标签（int64）。
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.classes = sorted(set(self.labels.tolist()))

        # 按类别建立样本索引
        self._class_indices: dict[int, np.ndarray] = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

    @classmethod
    def from_npz(cls, npz_path: Path) -> "FewShotDataset":
        """从第一章生成的窗口级 npz 文件加载。"""
        data = np.load(npz_path, allow_pickle=False)
        return cls(signals=data["signals"], labels=data["labels"])

    def n_classes(self) -> int:
        return len(self.classes)

    def n_samples_per_class(self) -> dict[int, int]:
        return {c: len(idx) for c, idx in self._class_indices.items()}

    def get_samples(self, class_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """从指定类别随机取 n 个样本，不放回。"""
        idx = self._class_indices[class_id]
        chosen = rng.choice(idx, size=n, replace=False)
        return self.signals[chosen]  # (n, L)


class EpisodeSampler:
    """元学习 episode 采样器。

    每次调用 sample() 返回一个完整 episode，其中标签已经过重映射。

    Parameters
    ----------
    dataset : FewShotDataset
    n_way : int
        每个 episode 包含的类别数（N-way）。
    k_shot : int
        支持集中每个类别的样本数（K-shot）。
    n_query : int
        查询集中每个类别的样本数。
    seed : int, optional
        随机种子，用于可复现实验。
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        seed: int | None = None,
    ) -> None:
        if n_way > dataset.n_classes():
            raise ValueError(
                f"n_way={n_way} exceeds available classes={dataset.n_classes()}"
            )
        min_samples = min(dataset.n_samples_per_class().values())
        if k_shot + n_query > min_samples:
            raise ValueError(
                f"k_shot+n_query={k_shot+n_query} exceeds min samples per class={min_samples}"
            )

        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Episode:
        """随机采样一个 episode，并完成标签重映射。

        重映射逻辑：
            随机选取 n_way 个全局类别，将它们按随机顺序映射为
            0, 1, ..., n_way-1，作为本次 episode 的临时标签。
            这一映射在 episode 结束后丢弃，下一个 episode 重新建立。
        """
        # 1. 随机选取 n_way 个类别，打乱顺序（随机标签重映射的来源）
        chosen_classes = self.rng.choice(
            self.dataset.classes, size=self.n_way, replace=False
        )
        self.rng.shuffle(chosen_classes)  # 打乱顺序增加多样性

        # 2. 建立标签重映射表：全局标签 → episode 内临时标签
        label_map: dict[int, int] = {
            int(c): i for i, c in enumerate(chosen_classes)
        }

        # 3. 为每个类别分别采样支持集和查询集
        support_x_list, support_y_list = [], []
        query_x_list, query_y_list = [], []

        for global_label in chosen_classes:
            local_label = label_map[int(global_label)]
            samples = self.dataset.get_samples(
                int(global_label), self.k_shot + self.n_query, self.rng
            )
            support_x_list.append(samples[: self.k_shot])
            query_x_list.append(samples[self.k_shot :])
            support_y_list.extend([local_label] * self.k_shot)
            query_y_list.extend([local_label] * self.n_query)

        # 4. 拼接并添加 channel 维度 (n, L) → (n, 1, L)
        support_x = np.stack(support_x_list, axis=0).reshape(-1, 1, self.dataset.signals.shape[-1])
        query_x   = np.stack(query_x_list,   axis=0).reshape(-1, 1, self.dataset.signals.shape[-1])
        support_y = np.array(support_y_list, dtype=np.int64)
        query_y   = np.array(query_y_list,   dtype=np.int64)

        return Episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            label_map=label_map,
        )

    def sample_batch(self, n_episodes: int) -> list[Episode]:
        """采样多个 episode，返回列表。"""
        return [self.sample() for _ in range(n_episodes)]
