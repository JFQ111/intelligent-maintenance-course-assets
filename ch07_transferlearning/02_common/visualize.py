"""Visualization helpers for Chapter 7 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    labels: list[str],
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tsne(features: np.ndarray, domains: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = TSNE(n_components=2, random_state=42, init="pca").fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=domains, cmap="tab10", s=10)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Domain", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
