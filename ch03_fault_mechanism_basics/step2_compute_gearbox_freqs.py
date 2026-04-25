"""第 3 章 任务二：计算齿轮箱啮合频率与边带频率。

用法：
    python step2_compute_gearbox_freqs.py --pinion-teeth 20 --gear-teeth 60 --pinion-rpm 1500 --output-dir ./outputs

脚本根据输入的齿轮参数计算 GMF、从动齿轮转频和边带频率，并打印结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from characteristic_freq_calculator import GearboxCharacteristicFrequencyCalculator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计算齿轮箱啮合频率与边带频率。"
    )
    parser.add_argument("--pinion-teeth", type=int, default=20, help="驱动齿轮齿数（默认 20）。")
    parser.add_argument("--gear-teeth", type=int, default=60, help="从动齿轮齿数（默认 60）。")
    parser.add_argument("--pinion-rpm", type=float, default=1500, help="驱动齿轮转速（rpm，默认 1500）。")
    parser.add_argument("--num-sidebands", type=int, default=3, help="边带阶数（默认 3）。")
    parser.add_argument("--output-dir", required=True, help="保存结果 CSV 的目录。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gb = GearboxCharacteristicFrequencyCalculator(
        pinion_teeth=args.pinion_teeth,
        gear_teeth=args.gear_teeth,
        pinion_rpm=args.pinion_rpm,
    )

    gmf = gb.calculate_gmf()
    sidebands = gb.get_sidebands(num_sidebands=args.num_sidebands)

    print("齿轮箱参数：")
    print(f"  驱动齿轮齿数：{args.pinion_teeth}")
    print(f"  从动齿轮齿数：{args.gear_teeth}")
    print(f"  传动比（从动/驱动）：{gb.gear_ratio():.4f}")
    print(f"  驱动齿轮转频：{gb.pinion_fr:.4f} Hz（{args.pinion_rpm} rpm）")
    print(f"  从动齿轮转频：{gb.gear_fr:.4f} Hz")
    print()
    print(f"齿轮啮合频率 GMF = {gmf:.4f} Hz")
    print()
    print(f"边带频率（以驱动齿轮转频 {gb.pinion_fr:.4f} Hz 为调制源）：")
    print(f"  下边带：{sidebands['lower']}")
    print(f"  GMF   ：{sidebands['gmf']}")
    print(f"  上边带：{sidebands['upper']}")

    # 保存边带表格
    rows = []
    for i, freq in enumerate(sidebands["lower"]):
        order = -(args.num_sidebands - i)
        rows.append({"order": order, "frequency_hz": freq, "type": "lower_sideband"})
    rows.append({"order": 0, "frequency_hz": sidebands["gmf"], "type": "GMF"})
    for i, freq in enumerate(sidebands["upper"]):
        rows.append({"order": i + 1, "frequency_hz": freq, "type": "upper_sideband"})

    df = pd.DataFrame(rows)
    csv_path = output_dir / "gearbox_sidebands.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
