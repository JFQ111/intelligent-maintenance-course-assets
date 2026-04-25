"""Pretrain the ADDA source encoder and classifier on the source domain."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def find_example_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if parent.name == "ch07_transfer_jnu":
            return parent
    raise RuntimeError("Cannot locate ch07_transfer_jnu root.")


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = find_example_root(SCRIPT_DIR)
COMMON_DIR = EXAMPLE_ROOT / "02_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from datasets import SingleDomainDataset
from models import ADDAModel
from train_utils import evaluate_classifier, get_device, save_metrics_csv, set_seed


def main() -> None:
    set_seed(42)
    device = get_device()

    processed_path = EXAMPLE_ROOT / "processed" / "da_600_to_1000.npz"
    results_dir = EXAMPLE_ROOT / "05_results"
    metrics_csv = results_dir / "metrics" / "adda_step1_metrics.csv"
    checkpoint_path = results_dir / "checkpoints" / "adda_source_pretrain.pt"
    results_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(processed_path, allow_pickle=True)
    source_dataset = SingleDomainDataset(
        X=data["X_source"],
        y=data["y_source"],
        d=data["d_source"],
    )
    target_test_dataset = SingleDomainDataset(
        X=data["X_target_test"],
        y=data["y_target_test"],
        d=data["d_target_test"],
    )

    source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
    source_eval_loader = DataLoader(source_dataset, batch_size=256, shuffle=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=256, shuffle=False)

    model = ADDAModel(feature_dim=128, num_classes=4).to(device)
    params = list(model.source_encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for x, y, _d in source_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model.predict_with_source(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        source_acc, _, _ = evaluate_classifier(
            model,
            source_eval_loader,
            device,
            predict_fn=lambda m, x: m.predict_with_source(x),
        )
        target_test_acc, _, _ = evaluate_classifier(
            model,
            target_test_loader,
            device,
            predict_fn=lambda m, x: m.predict_with_source(x),
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, num_batches),
            "source_acc": source_acc,
            "target_test_acc": target_test_acc,
        }
        save_metrics_csv(metrics_csv, row)
        print(
            f"[ADDA-Step1] epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"source_acc={source_acc:.4f} "
            f"target_test_acc={target_test_acc:.4f}"
        )

    torch.save(
        {
            "source_encoder": model.source_encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
