from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ALIGNED_LENGTH,
    FILE_SPECS,
    aligned_csv_name,
    aligned_npz_name,
    ensure_output_dir,
    expected_mat_path,
    label_names_array,
    labels_array,
    load_drive_end_signal,
    save_npz,
    truncate_signal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 CWRU 原始 .mat 文件转换为宽表 .csv 和长序列 .npz。")
    parser.add_argument("--data-root", required=True, help="存放 CWRU .mat 文件的目录。")
    parser.add_argument("--output-dir", required=True, help="保存转换结果的输出目录。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = ensure_output_dir(Path(args.output_dir))

    aligned_signals: list[np.ndarray] = []
    for spec in FILE_SPECS:
        file_path = expected_mat_path(data_root, spec.file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        signal, _, _ = load_drive_end_signal(file_path)
        aligned_signals.append(truncate_signal(signal, ALIGNED_LENGTH))

    signals = np.stack(aligned_signals, axis=0)
    label_names = label_names_array()
    labels = labels_array()

    csv_path = output_dir / aligned_csv_name()
    npz_path = output_dir / aligned_npz_name()

    wide_table = pd.DataFrame(signals.T, columns=label_names)
    wide_table.to_csv(csv_path, index=False)
    save_npz(npz_path, signals=signals, labels=labels, label_names=label_names)

    print(f"Aligned signals shape: {signals.shape}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved NPZ to: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
