"""提取 JNU 600 rpm 窗口样本的 25 个诊断特征。

输入:
    datasets/JNU/ 或 datatsets/JNU/ 下的 600 rpm CSV 文件

输出:
    outputs/jnu_600rpm_features.npz
    outputs/jnu_600rpm_features.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from feature_utils import FEATURE_NAMES, extract_feature_matrix


FS = 50_000
TARGET_SPEED = 600
WINDOW_SIZE = 1024
STRIDE = 1024
RANDOM_STATE = 42

FAULT_LABEL_MAP = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
LABEL_NAME_MAP = {
    "n": "正常",
    "ib": "内圈故障",
    "ob": "外圈故障",
    "tb": "滚动体故障",
}
FILE_PATTERN = re.compile(r"^(ib|ob|tb|n)600(?:_.*)?$", re.IGNORECASE)


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_example_root() -> Path:
    return Path(__file__).resolve().parent


def find_jnu_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "datasets" / "JNU",
        project_root / "datatsets" / "JNU",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 JNU 数据目录。请将 600 rpm 数据放到 datasets/JNU/ 或 datatsets/JNU/。"
    )


def iter_600rpm_files(jnu_dir: Path) -> Iterable[Path]:
    files = sorted(jnu_dir.glob("*.csv"))
    matched = [file_path for file_path in files if FILE_PATTERN.match(file_path.stem)]
    if not matched:
        raise FileNotFoundError("在 JNU 数据目录中未找到 600 rpm 的 CSV 文件。")
    return matched


def parse_fault_code(file_path: Path) -> str:
    match = FILE_PATTERN.match(file_path.stem)
    if match is None:
        raise ValueError(f"Unexpected file name: {file_path.name}")
    return match.group(1).lower()


def read_numeric_signal(file_path: Path) -> np.ndarray:
    table = pd.read_csv(file_path, header=None)
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce").dropna()
        if len(numeric) > 0:
            return numeric.to_numpy(dtype=np.float32)
    raise ValueError(f"{file_path.name} 中未找到数值列。")


def slice_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows: list[np.ndarray] = []
    for start in range(0, signal.shape[0] - window_size + 1, stride):
        window = signal[start : start + window_size]
        windows.append(window.astype(np.float32))
    if not windows:
        return np.empty((0, window_size), dtype=np.float32)
    return np.stack(windows, axis=0)


def main() -> None:
    project_root = find_project_root()
    example_root = find_example_root()
    output_dir = example_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    jnu_dir = find_jnu_dir(project_root)
    files = list(iter_600rpm_files(jnu_dir))
    rng = np.random.default_rng(RANDOM_STATE)

    window_bank: list[dict[str, object]] = []
    print(f"JNU data directory: {jnu_dir}")
    print(f"Sampling frequency: {FS} Hz")
    print(f"Window size: {WINDOW_SIZE}, stride: {STRIDE}")

    for file_path in files:
        fault_code = parse_fault_code(file_path)
        signal = read_numeric_signal(file_path)
        windows = slice_signal(signal, WINDOW_SIZE, STRIDE)
        if windows.shape[0] == 0:
            raise ValueError(f"{file_path.name} 的信号长度不足以切出窗口样本。")
        window_bank.append(
            {
                "file_name": file_path.name,
                "fault_code": fault_code,
                "label_id": FAULT_LABEL_MAP[fault_code],
                "label_name": LABEL_NAME_MAP[fault_code],
                "windows": windows,
            }
        )
        print(
            f"{file_path.name:<15} label={FAULT_LABEL_MAP[fault_code]} "
            f"name={LABEL_NAME_MAP[fault_code]:<6} windows={windows.shape[0]:>5} "
            f"window_shape={windows.shape}"
        )

    min_windows = min(int(item["windows"].shape[0]) for item in window_bank)
    print(f"\nBalanced windows per class: {min_windows}")

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    file_name_list: list[str] = []
    window_index_list: list[int] = []
    label_name_list: list[str] = []

    for item in sorted(window_bank, key=lambda row: int(row["label_id"])):
        windows = item["windows"]
        selected_indices = np.sort(rng.choice(windows.shape[0], size=min_windows, replace=False))
        selected_windows = windows[selected_indices]
        X_block = extract_feature_matrix(selected_windows, fs=FS)
        feature_blocks.append(X_block)
        label_blocks.append(np.full((min_windows,), int(item["label_id"]), dtype=np.int64))
        file_name_list.extend([str(item["file_name"])] * min_windows)
        window_index_list.extend(selected_indices.tolist())
        label_name_list.extend([str(item["label_name"])] * min_windows)
        print(
            f"Selected {min_windows:>4} windows from {item['file_name']:<15} "
            f"-> feature block shape={X_block.shape}"
        )

    X_features = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    y = np.concatenate(label_blocks, axis=0).astype(np.int64)
    label_names = np.array([LABEL_NAME_MAP[key] for key in sorted(FAULT_LABEL_MAP, key=FAULT_LABEL_MAP.get)], dtype=object)
    feature_names = np.array(FEATURE_NAMES, dtype=object)

    feature_df = pd.DataFrame(X_features, columns=FEATURE_NAMES)
    feature_df.insert(0, "label_name", label_name_list)
    feature_df.insert(0, "label_id", y)
    feature_df.insert(0, "window_index", window_index_list)
    feature_df.insert(0, "file_name", file_name_list)

    npz_path = output_dir / "jnu_600rpm_features.npz"
    csv_path = output_dir / "jnu_600rpm_features.csv"

    np.savez_compressed(
        npz_path,
        X_features=X_features,
        y=y,
        label_names=label_names,
        feature_names=feature_names,
        file_name=np.array(file_name_list, dtype=object),
        window_index=np.array(window_index_list, dtype=np.int64),
        fs=np.array([FS], dtype=np.int64),
        window_size=np.array([WINDOW_SIZE], dtype=np.int64),
        stride=np.array([STRIDE], dtype=np.int64),
    )
    feature_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    class_counts = pd.Series(y).value_counts().sort_index()
    print("\nFeature matrix summary:")
    print(f"  X_features.shape = {X_features.shape}")
    print(f"  y.shape          = {y.shape}")
    print(f"  class distribution:\n{class_counts.to_string()}")
    print(f"  feature count    = {len(FEATURE_NAMES)}")
    print(f"  feature names    = {FEATURE_NAMES}")
    print(f"Saved feature NPZ to: {npz_path}")
    print(f"Saved feature CSV to: {csv_path}")


if __name__ == "__main__":
    main()
