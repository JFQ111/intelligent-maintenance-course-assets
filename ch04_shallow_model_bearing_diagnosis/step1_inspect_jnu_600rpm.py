"""检查 JNU 600 rpm 原始数据文件。

输入:
    datasets/JNU/ 或 datatsets/JNU/ 下的 600 rpm CSV 文件

输出:
    examples/ch04_jnu_600rpm_shallow_diagnosis/outputs/jnu_600rpm_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_SPEED = 600
WINDOW_SIZE = 1024
STRIDE = 1024

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


def read_numeric_signal(file_path: Path) -> np.ndarray:
    table = pd.read_csv(file_path, header=None)
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce").dropna()
        if len(numeric) > 0:
            return numeric.to_numpy(dtype=np.float32)
    raise ValueError(f"{file_path.name} 中未找到数值列。")


def parse_fault_code(file_path: Path) -> str:
    match = FILE_PATTERN.match(file_path.stem)
    if match is None:
        raise ValueError(f"Unexpected file name: {file_path.name}")
    return match.group(1).lower()


def estimate_num_windows(signal_length: int, window_size: int, stride: int) -> int:
    if signal_length < window_size:
        return 0
    return 1 + (signal_length - window_size) // stride


def main() -> None:
    project_root = find_project_root()
    example_root = find_example_root()
    output_dir = example_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    jnu_dir = find_jnu_dir(project_root)
    files = list(iter_600rpm_files(jnu_dir))

    summary_rows: list[dict[str, object]] = []
    print(f"JNU data directory: {jnu_dir}")
    print(f"Inspecting {TARGET_SPEED} rpm files...")

    for file_path in files:
        fault_code = parse_fault_code(file_path)
        signal = read_numeric_signal(file_path)
        signal_length = int(signal.shape[0])
        estimated_windows = estimate_num_windows(signal_length, WINDOW_SIZE, STRIDE)
        row = {
            "file_name": file_path.name,
            "fault_code": fault_code,
            "label_name": LABEL_NAME_MAP[fault_code],
            "label_id": FAULT_LABEL_MAP[fault_code],
            "speed_rpm": TARGET_SPEED,
            "signal_length": signal_length,
            "signal_mean": float(np.mean(signal)),
            "signal_std": float(np.std(signal)),
            "signal_min": float(np.min(signal)),
            "signal_max": float(np.max(signal)),
            "estimated_windows": estimated_windows,
        }
        summary_rows.append(row)
        print(
            f"{file_path.name:<15} label={FAULT_LABEL_MAP[fault_code]} "
            f"name={LABEL_NAME_MAP[fault_code]:<6} length={signal_length:>8} "
            f"estimated_windows={estimated_windows:>5}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("label_id").reset_index(drop=True)
    output_path = output_dir / "jnu_600rpm_summary.csv"
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n类别映射关系:")
    for fault_code, label_id in FAULT_LABEL_MAP.items():
        print(f"  {fault_code:<2} -> {label_id} ({LABEL_NAME_MAP[fault_code]})")

    print("\n摘要表预览:")
    print(summary_df[["file_name", "label_name", "label_id", "signal_length", "estimated_windows"]].to_string(index=False))
    print(f"\nSaved summary CSV to: {output_path}")


if __name__ == "__main__":
    main()
