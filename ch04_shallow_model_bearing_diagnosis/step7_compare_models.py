"""汇总三种浅层模型的测试结果。

输入:
    outputs/decision_tree_metrics.csv
    outputs/svm_metrics.csv
    outputs/random_forest_metrics.csv

输出:
    outputs/model_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


METRIC_FILES = {
    "Decision Tree": "decision_tree_metrics.csv",
    "SVM": "svm_metrics.csv",
    "Random Forest": "random_forest_metrics.csv",
}


def find_example_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    example_root = find_example_root()
    output_dir = example_root / "outputs"
    output_path = output_dir / "model_comparison.csv"

    comparison_frames: list[pd.DataFrame] = []
    for model_name, file_name in METRIC_FILES.items():
        file_path = output_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"未找到指标文件: {file_path}。请先运行对应训练脚本。")
        metrics_df = pd.read_csv(file_path)
        metrics_df["Model"] = model_name
        comparison_frames.append(metrics_df)

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    comparison_df = comparison_df[
        [
            "Model",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
            "Validation Accuracy",
        ]
    ]
    comparison_df = comparison_df.sort_values("Accuracy", ascending=False, ignore_index=True)
    comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Model comparison:")
    print(comparison_df.to_string(index=False))
    print(f"Saved model comparison CSV to: {output_path}")


if __name__ == "__main__":
    main()
