"""Build DA and DG train/test splits from sliced JNU windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
SOURCE_DOMAIN_DA = 0   # 600 rpm
TARGET_DOMAIN_DA = 2   # 1000 rpm
SOURCE_DOMAINS_DG = (0, 1)  # 600, 800 rpm
TARGET_DOMAIN_DG = 2        # 1000 rpm


def find_example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    example_root = find_example_root()
    processed_dir = example_root / "processed"
    windows_path = processed_dir / "jnu_windows.npz"

    data = np.load(windows_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    domain = data["domain"].astype(np.int64)
    file_name = data["file_name"]

    source_mask_da = domain == SOURCE_DOMAIN_DA
    target_mask_da = domain == TARGET_DOMAIN_DA

    X_source = X[source_mask_da]
    y_source = y[source_mask_da]
    d_source = domain[source_mask_da]

    X_target = X[target_mask_da]
    y_target = y[target_mask_da]
    d_target = domain[target_mask_da]

    (
        X_target_train,
        X_target_test,
        y_target_train,
        y_target_test,
        d_target_train,
        d_target_test,
    ) = train_test_split(
        X_target,
        y_target,
        d_target,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_target,
    )

    da_output = processed_dir / "da_600_to_1000.npz"
    np.savez_compressed(
        da_output,
        X_source=X_source.astype(np.float32),
        y_source=y_source.astype(np.int64),
        d_source=d_source.astype(np.int64),
        X_target_train=X_target_train.astype(np.float32),
        d_target_train=d_target_train.astype(np.int64),
        X_target_test=X_target_test.astype(np.float32),
        y_target_test=y_target_test.astype(np.int64),
        d_target_test=d_target_test.astype(np.int64),
    )
    print(f"Saved DA split to: {da_output}")

    train_mask_dg = np.isin(domain, SOURCE_DOMAINS_DG)
    test_mask_dg = domain == TARGET_DOMAIN_DG

    dg_output = processed_dir / "dg_600_800_to_1000.npz"
    np.savez_compressed(
        dg_output,
        X_train=X[train_mask_dg].astype(np.float32),
        y_train=y[train_mask_dg].astype(np.int64),
        d_train=domain[train_mask_dg].astype(np.int64),
        X_test=X[test_mask_dg].astype(np.float32),
        y_test=y[test_mask_dg].astype(np.int64),
        d_test=domain[test_mask_dg].astype(np.int64),
        file_name_train=file_name[train_mask_dg],
        file_name_test=file_name[test_mask_dg],
    )
    print(f"Saved DG split to: {dg_output}")


if __name__ == "__main__":
    main()
