from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import FILE_SPECS, ensure_output_dir, expected_mat_path, load_drive_end_signal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 CWRU 原始 .mat 文件并输出摘要表。")
    parser.add_argument("--data-root", required=True, help="存放 CWRU .mat 文件的目录。")
    parser.add_argument("--output-dir", required=True, help="保存摘要表的输出目录。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = ensure_output_dir(Path(args.output_dir))

    records: list[dict[str, object]] = []
    for spec in FILE_SPECS:
        file_path = expected_mat_path(data_root, spec.file_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        signal, drive_end_key, rpm = load_drive_end_signal(file_path)
        records.append(
            {
                "file_id": spec.file_id,
                "label_id": spec.label_id,
                "label_name": spec.label_name,
                "description": spec.description,
                "mat_key": drive_end_key,
                "sample_count": len(signal),
                "dtype": str(signal.dtype),
                "mean": float(signal.mean()),
                "std": float(signal.std()),
                "min": float(signal.min()),
                "max": float(signal.max()),
                "rpm": rpm,
            }
        )

    summary = pd.DataFrame(records)
    summary_path = output_dir / "cwru_mat_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
