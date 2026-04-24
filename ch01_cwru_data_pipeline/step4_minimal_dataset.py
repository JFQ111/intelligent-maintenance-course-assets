from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_STEP_SIZE, DEFAULT_WINDOW_SIZE, windows_npz_name

try:
    from torch.utils.data import Dataset as BaseDataset
except Exception:
    class BaseDataset:
        pass


class CWRUWindowDataset(BaseDataset):
    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.label_names = data["label_names"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.signals[index], int(self.labels[index])

    def label_name(self, label_id: int) -> str:
        return str(self.label_names[label_id])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取窗口级 NPZ 并验证最小 Dataset 接口。")
    parser.add_argument("--data-root", required=True, help="保留为统一接口，当前脚本直接读取 output-dir 中的窗口级 .npz。")
    parser.add_argument("--output-dir", required=True, help="窗口级 .npz 所在目录。")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="窗口长度。")
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE, help="滑动步长。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    npz_path = Path(args.output_dir) / windows_npz_name(args.window_size, args.step_size)
    if not npz_path.exists():
        raise FileNotFoundError(f"Window NPZ not found: {npz_path}")

    dataset = CWRUWindowDataset(npz_path)
    first_signal, first_label = dataset[0]

    print(f"Dataset length: {len(dataset)}")
    print(f"First sample shape: {first_signal.shape}")
    print(f"First label id: {first_label}")
    print(f"First label name: {dataset.label_name(first_label)}")
    print("\nLabel mapping:")
    for label_id in range(len(dataset.label_names)):
        print(f"  {label_id}: {dataset.label_name(label_id)}")

    print("\nPreview of the first three samples:")
    for index in range(3):
        signal, label = dataset[index]
        print(
            f"  index={index}, label={label}, label_name={dataset.label_name(label)}, "
            f"mean={signal.mean():.4f}, std={signal.std():.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
