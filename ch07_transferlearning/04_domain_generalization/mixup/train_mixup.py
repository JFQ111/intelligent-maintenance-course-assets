"""Train a Mixup-based DG baseline on JNU source domains 600 and 800."""

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

from datasets import MultiSourceDataset, SingleDomainDataset
from losses import mixup_data, mixup_loss
from models import BaseClassifier
from train_utils import evaluate_classifier, get_device, save_metrics_csv, set_seed


def main() -> None:
    set_seed(42)
    device = get_device()

    processed_path = EXAMPLE_ROOT / "processed" / "dg_600_800_to_1000.npz"
    results_dir = EXAMPLE_ROOT / "05_results"
    metrics_csv = results_dir / "metrics" / "mixup_metrics.csv"
    checkpoint_path = results_dir / "checkpoints" / "mixup_model.pt"
    results_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(processed_path, allow_pickle=True)
    train_dataset = MultiSourceDataset(
        X=data["X_train"],
        y=data["y_train"],
        d=data["d_train"],
    )
    test_dataset = SingleDomainDataset(
        X=data["X_test"],
        y=data["y_test"],
        d=data["d_test"],
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = BaseClassifier(feature_dim=128, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for x, y, _d in train_loader:
            x = x.to(device)
            y = y.to(device)
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            logits = model(mixed_x)
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        source_acc, _, _ = evaluate_classifier(model, train_eval_loader, device)
        unseen_target_acc, _, _ = evaluate_classifier(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, num_batches),
            "source_acc": source_acc,
            "unseen_target_acc": unseen_target_acc,
        }
        save_metrics_csv(metrics_csv, row)
        print(
            f"[DG-Mixup] epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"source_acc={source_acc:.4f} "
            f"unseen_target_acc={unseen_target_acc:.4f}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
