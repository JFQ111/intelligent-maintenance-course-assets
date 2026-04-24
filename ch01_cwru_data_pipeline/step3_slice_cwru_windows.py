from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DEFAULT_STEP_SIZE,
    DEFAULT_WINDOW_SIZE,
    aligned_npz_name,
    ensure_output_dir,
    save_npz,
    slice_signal,
    windows_npz_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 CWRU 长序列执行滑动窗口切片。")
    parser.add_argument("--data-root", required=True, help="保留为统一接口，当前脚本直接读取 output-dir 中的长序列 .npz。")
    parser.add_argument("--output-dir", required=True, help="长序列 .npz 与窗口级 .npz 的目录。")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="窗口长度。")
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE, help="滑动步长。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))

    aligned_npz_path = output_dir / aligned_npz_name()
    if not aligned_npz_path.exists():
        raise FileNotFoundError(f"Aligned NPZ not found: {aligned_npz_path}")

    data = np.load(aligned_npz_path, allow_pickle=False)
    raw_signals = data["signals"]
    raw_labels = data["labels"]
    label_names = data["label_names"]

    window_signals: list[np.ndarray] = []
    window_labels: list[np.ndarray] = []
    summary_rows: list[dict[str, object]] = []

    for signal, label in zip(raw_signals, raw_labels):
        label = int(label)
        windows = slice_signal(signal, args.window_size, args.step_size)
        window_signals.append(windows)
        window_labels.append(np.full(len(windows), label, dtype=np.int64))
        summary_rows.append(
            {
                "label_id": label,
                "label_name": str(label_names[label]),
                "window_count": len(windows),
                "window_size": args.window_size,
                "step_size": args.step_size,
            }
        )

    signals = np.concatenate(window_signals, axis=0)
    labels = np.concatenate(window_labels, axis=0)

    output_npz_path = output_dir / windows_npz_name(args.window_size, args.step_size)
    save_npz(output_npz_path, signals=signals, labels=labels, label_names=label_names)

    summary_path = output_dir / f"cwru_window_summary_w{args.window_size}_s{args.step_size}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Window signals shape: {signals.shape}")
    print(f"Window labels shape: {labels.shape}")
    print(f"Saved window NPZ to: {output_npz_path}")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
