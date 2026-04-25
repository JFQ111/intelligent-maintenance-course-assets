"""Inspect JNU CSV/Excel files and summarize fault/domain metadata.

This script scans datasets/JNU (or datatsets/JNU as a fallback), parses each
file name, reads the raw table shape, and writes a summary CSV for teaching use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

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


def main() -> None:
    project_root = find_project_root()
    example_root = find_example_root()
    jnu_dir = find_jnu_dir(project_root)
    metrics_dir = example_root / "05_results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    print(f"Scanning JNU files from: {jnu_dir}")

    for file_path in iter_data_files(jnu_dir):
        fault_type, speed = parse_file_name(file_path)
        table = read_table(file_path)
        row = {
            "file_name": file_path.name,
            "rows": int(table.shape[0]),
            "cols": int(table.shape[1]),
            "fault_type": fault_type,
            "fault_label": FAULT_MAP[fault_type],
            "speed": speed,
            "domain_label": DOMAIN_MAP[speed],
        }
        summary_rows.append(row)
        print(
            f"{file_path.name:<18} rows={table.shape[0]:>7} cols={table.shape[1]:>3} "
            f"fault={fault_type:<2} speed={speed} domain={DOMAIN_MAP[speed]}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["domain_label", "fault_label", "file_name"])
    output_csv = metrics_dir / "jnu_file_summary.csv"
    summary_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV to: {output_csv}")


if __name__ == "__main__":
    main()
