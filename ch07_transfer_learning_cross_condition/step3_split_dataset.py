"""划分 JNU 600 rpm 特征数据集。

输入:
    outputs/jnu_600rpm_features.npz

输出:
    outputs/jnu_600rpm_split.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def find_example_root() -> Path:
    return Path(__file__).resolve().parent


def print_class_distribution(name: str, labels: np.ndarray) -> None:
    distribution = pd.Series(labels).value_counts().sort_index()
    print(f"{name} class distribution:")
    print(distribution.to_string())


def main() -> None:
    example_root = find_example_root()
    output_dir = example_root / "outputs"
    input_path = output_dir / "jnu_600rpm_features.npz"
    output_path = output_dir / "jnu_600rpm_split.npz"

    if not input_path.exists():
        raise FileNotFoundError(f"未找到特征文件: {input_path}。请先运行 step2_extract_25_features.py")

    data = np.load(input_path, allow_pickle=True)
    X_features = data["X_features"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = data["feature_names"]
    label_names = data["label_names"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features,
        y,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    np.savez_compressed(
        output_path,
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        label_names=label_names,
    )

    print("Split summary:")
    print(f"  X_train.shape = {X_train.shape}")
    print(f"  X_val.shape   = {X_val.shape}")
    print(f"  X_test.shape  = {X_test.shape}")
    print(f"  y_train.shape = {y_train.shape}")
    print(f"  y_val.shape   = {y_val.shape}")
    print(f"  y_test.shape  = {y_test.shape}")
    print_class_distribution("Train", y_train)
    print_class_distribution("Validation", y_val)
    print_class_distribution("Test", y_test)
    print(f"Saved split NPZ to: {output_path}")


if __name__ == "__main__":
    main()
