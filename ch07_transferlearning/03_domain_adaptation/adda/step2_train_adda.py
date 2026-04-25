"""Train ADDA target encoder with an adversarial discriminator."""

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
from models import ADDAModel
from train_utils import evaluate_classifier, get_device, save_metrics_csv, set_seed


def main() -> None:
    set_seed(42)
    device = get_device()

    processed_path = EXAMPLE_ROOT / "processed" / "da_600_to_1000.npz"
    results_dir = EXAMPLE_ROOT / "05_results"
    metrics_csv = results_dir / "metrics" / "adda_step2_metrics.csv"
    source_checkpoint = results_dir / "checkpoints" / "adda_source_pretrain.pt"
    output_checkpoint = results_dir / "checkpoints" / "adda_target_encoder.pt"

    data = np.load(processed_path, allow_pickle=True)
    train_dataset = DomainAdaptationDataset(
        X_source=data["X_source"],
        y_source=data["y_source"],
        d_source=data["d_source"],
        X_target=data["X_target_train"],
        d_target=data["d_target_train"],
    )
    source_eval_dataset = SingleDomainDataset(
        X=data["X_source"],
        y=data["y_source"],
        d=data["d_source"],
    )
    target_test_dataset = SingleDomainDataset(
        X=data["X_target_test"],
        y=data["y_target_test"],
        d=data["d_target_test"],
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    source_eval_loader = DataLoader(source_eval_dataset, batch_size=256, shuffle=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=256, shuffle=False)

    model = ADDAModel(feature_dim=128, num_classes=4).to(device)
    checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=True)
    model.source_encoder.load_state_dict(checkpoint["source_encoder"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    model.target_encoder.load_state_dict(checkpoint["source_encoder"])

    for param in model.source_encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False

    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    optimizer_target = torch.optim.Adam(model.target_encoder.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        running_disc = 0.0
        running_adv = 0.0
        num_batches = 0

        for x_s, _y_s, _d_s, x_t, _d_t in train_loader:
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            source_domain = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
            target_domain = torch.ones(x_t.size(0), dtype=torch.long, device=device)

            feat_s = model.source_encoder(x_s).detach()
            feat_t = model.target_encoder(x_t).detach()
            logits_s = model.discriminator(feat_s)
            logits_t = model.discriminator(feat_t)
            disc_logits = torch.cat([logits_s, logits_t], dim=0)
            disc_labels = torch.cat([source_domain, target_domain], dim=0)
            loss_disc = criterion(disc_logits, disc_labels)

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            feat_t = model.target_encoder(x_t)
            logits_t = model.discriminator(feat_t)
            fool_labels = torch.zeros(x_t.size(0), dtype=torch.long, device=device)
            loss_adv = criterion(logits_t, fool_labels)

            optimizer_target.zero_grad()
            loss_adv.backward()
            optimizer_target.step()

            running_disc += float(loss_disc.item())
            running_adv += float(loss_adv.item())
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
            predict_fn=lambda m, x: m.predict_with_target(x),
        )
        row = {
            "epoch": epoch,
            "loss_disc": running_disc / max(1, num_batches),
            "loss_adv": running_adv / max(1, num_batches),
            "source_acc": source_acc,
            "target_test_acc": target_test_acc,
        }
        save_metrics_csv(metrics_csv, row)
        print(
            f"[ADDA-Step2] epoch={epoch:02d} "
            f"loss_disc={row['loss_disc']:.4f} "
            f"loss_adv={row['loss_adv']:.4f} "
            f"source_acc={source_acc:.4f} "
            f"target_test_acc={target_test_acc:.4f}"
        )

    torch.save(
        {
            "target_encoder": model.target_encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "discriminator": model.discriminator.state_dict(),
        },
        output_checkpoint,
    )
    print(f"Saved checkpoint to: {output_checkpoint}")


if __name__ == "__main__":
    main()
