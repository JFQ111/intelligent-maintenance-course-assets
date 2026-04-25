"""Shared training utilities for Chapter 7 experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    predict_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate a classifier on a labeled loader.

    Returns:
        acc: float
        y_true: [N]
        y_pred: [N]
    """

    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)

        logits = predict_fn(model, x) if predict_fn is not None else model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        pred = logits.argmax(dim=1)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=np.int64)
    acc_value = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    return acc_value, y_true, y_pred


def save_metrics_csv(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
