"""Dataset wrappers used by Chapter 7 DA/DG training scripts.

Shapes:
    X: [N, 1, 1024] float32
    y: [N] int64
    d: [N] int64
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype)


class SingleDomainDataset(Dataset):
    """A standard tensor-style dataset for one labeled domain."""

    def __init__(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> None:
        self.X = _to_tensor(X, torch.float32)
        self.y = _to_tensor(y, torch.long)
        self.d = _to_tensor(d, torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.d[index]


class DomainAdaptationDataset(Dataset):
    """Return paired source and target samples for DA training.

    Returns:
        x_s: [1, 1024]
        y_s: []
        d_s: []
        x_t: [1, 1024]
        d_t: []
    """

    def __init__(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        d_source: np.ndarray,
        X_target: np.ndarray,
        d_target: np.ndarray,
    ) -> None:
        self.X_source = _to_tensor(X_source, torch.float32)
        self.y_source = _to_tensor(y_source, torch.long)
        self.d_source = _to_tensor(d_source, torch.long)
        self.X_target = _to_tensor(X_target, torch.float32)
        self.d_target = _to_tensor(d_target, torch.long)

    def __len__(self) -> int:
        return int(self.X_source.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        target_index = index % len(self.X_target)
        return (
            self.X_source[index],
            self.y_source[index],
            self.d_source[index],
            self.X_target[target_index],
            self.d_target[target_index],
        )


class MultiSourceDataset(Dataset):
    """A labeled multi-domain dataset used by DG training scripts."""

    def __init__(self, X: np.ndarray, y: np.ndarray, d: np.ndarray) -> None:
        self.X = _to_tensor(X, torch.float32)
        self.y = _to_tensor(y, torch.long)
        self.d = _to_tensor(d, torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.d[index]
