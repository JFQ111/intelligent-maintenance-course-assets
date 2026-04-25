"""Train a simple DANN baseline on the JNU DA split."""

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

from datasets import DomainAdaptationDataset, SingleDomainDataset
from models import DANNModel
from train_utils import evaluate_classifier, get_device, save_metrics_csv, set_seed


def main() -> None:
    set_seed(42)
    device = get_device()

    processed_path = EXAMPLE_ROOT / "processed" / "da_600_to_1000.npz"
    results_dir = EXAMPLE_ROOT / "05_results"
    metrics_csv = results_dir / "metrics" / "dann_metrics.csv"
    checkpoint_path = results_dir / "checkpoints" / "dann_model.pt"
    results_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    results_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(processed_path, allow_pickle=True)
    train_dataset = DomainAdaptationDataset(
        X_source=data["X_source"],
        y_source=data["y_source"],
        d_source=data["d_source"],
        X_target=data["X_target_train"],
        d_target=data["d_target_train"],
    )
    target_test_dataset = SingleDomainDataset(
        X=data["X_target_test"],
        y=data["y_target_test"],
        d=data["d_target_test"],
    )
    source_eval_dataset = SingleDomainDataset(
        X=data["X_source"],
        y=data["y_source"],
        d=data["d_source"],
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=256, shuffle=False)
    source_eval_loader = DataLoader(source_eval_dataset, batch_size=256, shuffle=False)

    model = DANNModel(feature_dim=128, num_classes=4, num_domains=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    epochs = 3
    lambda_adv = 0.2

    for epoch in range(1, epochs + 1):
        model.train()
        running_cls = 0.0
        running_domain = 0.0
        running_total = 0.0
        num_batches = 0

        for x_s, y_s, _d_s, x_t, _d_t in train_loader:
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            domain_s = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
            domain_t = torch.ones(x_t.size(0), dtype=torch.long, device=device)

            class_logits_s, domain_logits_s, _ = model(x_s, lambda_grl=lambda_adv)
            _, domain_logits_t, _ = model(x_t, lambda_grl=lambda_adv)

            loss_cls = criterion_cls(class_logits_s, y_s)
            domain_logits = torch.cat([domain_logits_s, domain_logits_t], dim=0)
            domain_labels = torch.cat([domain_s, domain_t], dim=0)
            loss_domain = criterion_domain(domain_logits, domain_labels)
            total_loss = loss_cls + lambda_adv * loss_domain

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_cls += float(loss_cls.item())
            running_domain += float(loss_domain.item())
            running_total += float(total_loss.item())
            num_batches += 1

        source_acc, _, _ = evaluate_classifier(
            model,
            source_eval_loader,
            device,
            predict_fn=lambda m, x: m.predict(x),
        )
        target_test_acc, _, _ = evaluate_classifier(
            model,
            target_test_loader,
            device,
            predict_fn=lambda m, x: m.predict(x),
        )

        row = {
            "epoch": epoch,
            "loss_cls": running_cls / max(1, num_batches),
            "loss_domain": running_domain / max(1, num_batches),
            "total_loss": running_total / max(1, num_batches),
            "source_acc": source_acc,
            "target_test_acc": target_test_acc,
        }
        save_metrics_csv(metrics_csv, row)
        print(
            f"[DANN] epoch={epoch:02d} "
            f"loss_cls={row['loss_cls']:.4f} "
            f"loss_domain={row['loss_domain']:.4f} "
            f"total_loss={row['total_loss']:.4f} "
            f"source_acc={source_acc:.4f} "
            f"target_test_acc={target_test_acc:.4f}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
