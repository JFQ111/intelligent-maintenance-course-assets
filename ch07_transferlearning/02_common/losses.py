"""Loss functions used in Chapter 7 DA/DG experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute an RBF kernel matrix.

    Args:
        x: [B1, F]
        y: [B2, F]
    Returns:
        kernel: [B1, B2]
    """

    pairwise_dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-pairwise_dist / (2.0 * sigma * sigma))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...] = (1.0, 5.0, 10.0)) -> torch.Tensor:
    """Compute MMD loss between source and target features.

    Args:
        x: [B, F] source features
        y: [B, F] target features
    Returns:
        scalar MMD loss
    """

    if x.size(0) == 0 or y.size(0) == 0:
        return torch.zeros((), device=x.device if x.numel() else y.device)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for sigma in sigmas:
        k_xx = k_xx + _gaussian_kernel(x, x, sigma).mean()
        k_yy = k_yy + _gaussian_kernel(y, y, sigma).mean()
        k_xy = k_xy + _gaussian_kernel(x, y, sigma).mean()
    return k_xx + k_yy - 2.0 * k_xy


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss on two feature batches.

    Args:
        source: [B1, F]
        target: [B2, F]
    Returns:
        scalar CORAL loss
    """

    if source.size(0) < 2 or target.size(0) < 2:
        return torch.zeros((), device=source.device)

    d = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cov_source = (source_centered.t() @ source_centered) / (source.size(0) - 1)
    cov_target = (target_centered.t() @ target_centered) / (target.size(0) - 1)
    return ((cov_source - cov_target) ** 2).sum() / (4.0 * d * d)


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup on a batch.

    Args:
        x: [B, C, L]
        y: [B]
    Returns:
        mixed_x: [B, C, L]
        y_a: [B]
        y_b: [B]
        lam: scalar mixing coefficient
    """

    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_loss(
    criterion: torch.nn.Module,
    predictions: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute Mixup classification loss.

    Args:
        predictions: [B, num_classes]
        y_a: [B]
        y_b: [B]
    """

    return lam * criterion(predictions, y_a) + (1.0 - lam) * criterion(predictions, y_b)
