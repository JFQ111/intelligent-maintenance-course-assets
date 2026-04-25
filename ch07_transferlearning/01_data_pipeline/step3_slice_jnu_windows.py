"""Slice JNU long signals into fixed windows.

Input:
    processed/jnu_long_signals.npz

Output:
    processed/jnu_windows.npz

Saved arrays:
    X: float32 array with shape [N, 1, 1024]
    y: int64 array with shape [N]
    domain: int64 array with shape [N]
    file_name: object array with shape [N]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


WINDOW_SIZE = 1024
STRIDE = 512


def find_example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def slice_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows: list[np.ndarray] = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start : start + window_size]
        windows.append(window.astype(np.float32))
    if not windows:
        return np.empty((0, window_size), dtype=np.float32)
    return np.stack(windows, axis=0)


def main() -> None:
    example_root = find_example_root()
    processed_dir = example_root / "processed"
    input_path = processed_dir / "jnu_long_signals.npz"
    output_path = processed_dir / "jnu_windows.npz"

    data = np.load(input_path, allow_pickle=True)
    signals = data["signal"]
    fault_labels = data["fault_label"].astype(np.int64)
    domain_labels = data["domain_label"].astype(np.int64)
    file_names = data["file_name"]

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_domain: list[np.ndarray] = []
    all_file_name: list[np.ndarray] = []

    for signal, fault_label, domain_label, file_name in zip(signals, fault_labels, domain_labels, file_names):
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        windows = slice_signal(signal, WINDOW_SIZE, STRIDE)
        if len(windows) == 0:
            continue

        all_x.append(windows[:, None, :])
        all_y.append(np.full((len(windows),), int(fault_label), dtype=np.int64))
        all_domain.append(np.full((len(windows),), int(domain_label), dtype=np.int64))
        all_file_name.append(np.full((len(windows),), str(file_name), dtype=object))

        print(
            f"Sliced {file_name:<18} signal_len={len(signal):>8} "
            f"windows={len(windows):>6} fault={fault_label} domain={domain_label}"
        )

    X = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    domain = np.concatenate(all_domain, axis=0).astype(np.int64)
    file_name = np.concatenate(all_file_name, axis=0)

    np.savez_compressed(output_path, X=X, y=y, domain=domain, file_name=file_name)
    print(f"Saved windows NPZ to: {output_path}")
    print(f"X shape={X.shape}, y shape={y.shape}, domain shape={domain.shape}")


if __name__ == "__main__":
    main()
