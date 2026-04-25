"""第 3 章 任务三：核对 CWRU 10 类子集各工况下的轴承特征频率。

用法：
    python step3_verify_cwru_freqs.py --output-dir ./outputs

CWRU 数据集提供了四个负载工况，对应四个近似转速（rpm）：
    0 hp → ~1797 rpm
    1 hp → ~1772 rpm
    2 hp → ~1750 rpm
    3 hp → ~1730 rpm

本脚本对每个转速计算 BPFO、BPFI、BSF、CF，并与教材/文献中的参考值对比，
输出差值（Hz），以验证 characteristic_freq_calculator 的计算正确性。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from characteristic_freq_calculator import (
    BearingParameters,
    BearingCharacteristicFrequencyCalculator,
)

# CWRU 6205-2RS JEM SKF 轴承标准参数
CWRU_BEARING = BearingParameters(N=9, d=7.94, D=38.5, alpha=0.0)

# CWRU 文献中常引用的参考特征频率（1797 rpm）
# 来源：Randall & Antoni (2011), Table 1
REFERENCE_1797 = {
    "BPFO": 107.36,
    "BPFI": 162.18,
    "BSF":  71.42,
    "CF":   11.93,
}

# CWRU 四个负载工况的近似转速
CWRU_OPERATING_CONDITIONS = [
    {"load_hp": 0, "rpm": 1797},
    {"load_hp": 1, "rpm": 1772},
    {"load_hp": 2, "rpm": 1750},
    {"load_hp": 3, "rpm": 1730},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="核对 CWRU 轴承特征频率与文献参考值。"
    )
    parser.add_argument("--output-dir", required=True, help="保存结果 CSV 的目录。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator = BearingCharacteristicFrequencyCalculator(CWRU_BEARING)

    print("=" * 60)
    print("CWRU 6205-2RS JEM SKF 轴承特征频率计算结果")
    print("=" * 60)

    all_rows = []
    for condition in CWRU_OPERATING_CONDITIONS:
        rpm = condition["rpm"]
        freqs = calculator.calculate_all_frequencies(rpm)
        row = {"load_hp": condition["load_hp"], "rpm": rpm}
        row.update({k: round(v, 2) for k, v in freqs.items()})
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(df.to_string(index=False))

    # 与 1797 rpm 参考值对比
    print()
    print("=" * 60)
    print("1797 rpm 下与文献参考值（Randall & Antoni, 2011）的对比")
    print("=" * 60)

    computed_1797 = calculator.calculate_all_frequencies(1797)
    ref_rows = []
    for key in ["BPFO", "BPFI", "BSF", "CF"]:
        computed = round(computed_1797[key], 2)
        ref = REFERENCE_1797[key]
        diff = round(computed - ref, 2)
        ref_rows.append({"频率类型": key, "计算值 (Hz)": computed, "参考值 (Hz)": ref, "差值 (Hz)": diff})

    df_ref = pd.DataFrame(ref_rows)
    print(df_ref.to_string(index=False))

    # 保存两张表
    csv_all = output_dir / "cwru_bearing_freqs_all_conditions.csv"
    csv_ref = output_dir / "cwru_bearing_freqs_reference_check.csv"
    df.to_csv(csv_all, index=False)
    df_ref.to_csv(csv_ref, index=False, encoding="utf-8-sig")

    print(f"\nSaved to: {csv_all}")
    print(f"Saved to: {csv_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
