"""Check the generated DA and DG dataset files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


def find_example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def describe_distribution(name: str, values: np.ndarray) -> str:
    counter = Counter(values.tolist())
    return f"{name}={dict(sorted(counter.items()))}"


def inspect_npz(npz_path: Path) -> None:
    print("=" * 100)
    print(f"Inspecting: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray):
            print(f"{key:<16} shape={value.shape} dtype={value.dtype}")
            if key.startswith("y"):
                print("  " + describe_distribution("class_dist", value.astype(np.int64)))
            if key.startswith("d"):
                print("  " + describe_distribution("domain_dist", value.astype(np.int64)))
        else:
            print(f"{key:<16} type={type(value)}")


def main() -> None:
    example_root = find_example_root()
    processed_dir = example_root / "processed"
    inspect_npz(processed_dir / "da_600_to_1000.npz")
    inspect_npz(processed_dir / "dg_600_800_to_1000.npz")


if __name__ == "__main__":
    main()
