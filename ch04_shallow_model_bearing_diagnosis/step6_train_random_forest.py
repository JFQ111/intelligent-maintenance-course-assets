"""训练 JNU 600 rpm 随机森林分类器。

输入:
    outputs/jnu_600rpm_split.npz

输出:
    outputs/random_forest_metrics.csv
    outputs/random_forest_confusion_matrix.csv
    outputs/random_forest_feature_importance.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RANDOM_STATE = 42


def find_example_root() -> Path:
    return Path(__file__).resolve().parent


def build_metrics_row(
    model_name: str,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> pd.DataFrame:
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    row = {
        "Model": model_name,
        "Validation Accuracy": float(accuracy_score(y_val, y_val_pred)),
        "Accuracy": float(accuracy_score(y_test, y_test_pred)),
        "Macro Precision": float(report_dict["macro avg"]["precision"]),
        "Macro Recall": float(report_dict["macro avg"]["recall"]),
        "Macro F1": float(report_dict["macro avg"]["f1-score"]),
        "Weighted Precision": float(report_dict["weighted avg"]["precision"]),
        "Weighted Recall": float(report_dict["weighted avg"]["recall"]),
        "Weighted F1": float(report_dict["weighted avg"]["f1-score"]),
    }
    return pd.DataFrame([row])


def main() -> None:
    example_root = find_example_root()
    output_dir = example_root / "outputs"
    input_path = output_dir / "jnu_600rpm_split.npz"
    metrics_path = output_dir / "random_forest_metrics.csv"
    confusion_path = output_dir / "random_forest_confusion_matrix.csv"
    importance_path = output_dir / "random_forest_feature_importance.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"未找到划分文件: {input_path}。请先运行 step3_split_dataset.py")

    data = np.load(input_path, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_val = data["y_val"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    label_names = [str(item) for item in data["label_names"].tolist()]
    feature_names = [str(item) for item in data["feature_names"].tolist()]

    print("Training Random Forest...")
    print(f"  X_train.shape = {X_train.shape}")
    print(f"  X_val.shape   = {X_val.shape}")
    print(f"  X_test.shape  = {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    report_text = classification_report(y_test, y_test_pred, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(len(label_names))))

    metrics_df = build_metrics_row("Random Forest", y_val, y_val_pred, y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.index.name = "True Label"
    cm_df.columns.name = "Predicted Label"

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    cm_df.to_csv(confusion_path, encoding="utf-8-sig")
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print("Classification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm_df.to_string())
    print("Top 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {confusion_path}")
    print(f"Saved feature importance to: {importance_path}")


if __name__ == "__main__":
    main()
