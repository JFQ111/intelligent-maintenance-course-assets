"""第 3 章 任务一：计算轴承特征频率并输出摘要表。

用法：
    python step1_compute_bearing_freqs.py --rpm 1797 --output-dir ./outputs

脚本使用 CWRU 6205-2RS JEM SKF 轴承的标准参数，在指定转速下计算
BPFO、BPFI、BSF 和 CF，并以表格形式打印，同时保存为 CSV 文件。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from characteristic_freq_calculator import (
    BearingParameters,
    BearingCharacteristicFrequencyCalculator,
)

# CWRU 6205-2RS JEM SKF 轴承参数（N=9, d=7.94 mm, D=38.5 mm, alpha=0°）
CWRU_BEARING = BearingParameters(N=9, d=7.94, D=38.5, alpha=0.0)

# CWRU 数据集中常用的四个转速（rpm）
CWRU_RPM_LIST = [1797, 1772, 1750, 1730]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算 CWRU 轴承特征频率并输出摘要表。"
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="指定单个转速（rpm）。若不指定，则遍历 CWRU 常用的四个转速。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="保存结果 CSV 的目录。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator = BearingCharacteristicFrequencyCalculator(CWRU_BEARING)
    rpm_list = [args.rpm] if args.rpm is not None else CWRU_RPM_LIST

    print("轴承参数：")
    print(
        f"  N={CWRU_BEARING.N}, d={CWRU_BEARING.d} mm, "
        f"D={CWRU_BEARING.D} mm, alpha={CWRU_BEARING.alpha}°"
    )
    print()

    records = []
    for rpm in rpm_list:
        freqs = calculator.calculate_all_frequencies(rpm)
        records.append({"rpm": rpm, **{k: round(v, 4) for k, v in freqs.items()}})

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    csv_path = output_dir / "bearing_characteristic_freqs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
