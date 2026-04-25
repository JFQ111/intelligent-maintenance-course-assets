"""Convert raw JNU CSV/Excel files into a unified long-signal NPZ file.

Output file:
    examples/ch07_transfer_jnu/processed/jnu_long_signals.npz

Saved arrays:
    signal: object array, each item is a 1D float32 signal
    fault_label: int64 array of shape [num_files]
    domain_label: int64 array of shape [num_files]
    file_name: object array of shape [num_files]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FAULT_MAP = {"n": 0, "ib": 1, "ob": 2, "tb": 3}
DOMAIN_MAP = {600: 0, 800: 1, 1000: 2}
FILE_PATTERN = re.compile(r"^(ib|ob|tb|n)(600|800|1000)(?:_.*)?$", re.IGNORECASE)


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_jnu_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "datasets" / "JNU",
        project_root / "datatsets" / "JNU",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find datasets/JNU or datatsets/JNU.")


def iter_data_files(jnu_dir: Path) -> Iterable[Path]:
    patterns = ("*.csv", "*.xlsx", "*.xls")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(jnu_dir.glob(pattern)))
    return files


def parse_file_name(file_path: Path) -> tuple[str, int]:
    match = FILE_PATTERN.match(file_path.stem)
    if match is None:
        raise ValueError(f"Unexpected file name format: {file_path.name}")
    fault_type = match.group(1).lower()
    speed = int(match.group(2))
    return fault_type, speed


def read_table(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, header=None)
    return pd.read_excel(file_path, header=None)


def select_first_numeric_series(table: pd.DataFrame) -> np.ndarray:
    """Return the first numeric column as a 1D float32 array.

    The raw JNU files are currently single-column CSV files, but this function
    also supports multi-column CSV/Excel input by skipping non-numeric columns.
    """

    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce").dropna()
        if len(numeric) == 0:
            continue
        return numeric.to_numpy(dtype=np.float32)
    raise ValueError("No numeric column found in the file.")


def main() -> None:
    project_root = find_project_root()
    example_root = find_example_root()
    jnu_dir = find_jnu_dir(project_root)
    processed_dir = example_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    signals: list[np.ndarray] = []
    fault_labels: list[int] = []
    domain_labels: list[int] = []
    file_names: list[str] = []

    for file_path in iter_data_files(jnu_dir):
        fault_type, speed = parse_file_name(file_path)
        table = read_table(file_path)
        signal = select_first_numeric_series(table)

        signals.append(signal.astype(np.float32))
        fault_labels.append(FAULT_MAP[fault_type])
        domain_labels.append(DOMAIN_MAP[speed])
        file_names.append(file_path.name)

        print(
            f"Loaded {file_path.name:<18} "
            f"len={len(signal):>8} fault={FAULT_MAP[fault_type]} domain={DOMAIN_MAP[speed]}"
        )

    output_path = processed_dir / "jnu_long_signals.npz"
    np.savez_compressed(
        output_path,
        signal=np.array(signals, dtype=object),
        fault_label=np.array(fault_labels, dtype=np.int64),
        domain_label=np.array(domain_labels, dtype=np.int64),
        file_name=np.array(file_names, dtype=object),
    )
    print(f"Saved long-signal NPZ to: {output_path}")


if __name__ == "__main__":
    main()
