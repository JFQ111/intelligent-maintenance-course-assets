from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
ALIGNED_LENGTH = 119_808
DEFAULT_WINDOW_SIZE = 1_024
DEFAULT_STEP_SIZE = 512


@dataclass(frozen=True)
class CWRUFileSpec:
    file_id: str
    label_id: int
    label_name: str
    description: str


FILE_SPECS: tuple[CWRUFileSpec, ...] = (
    CWRUFileSpec("097", 0, "normal", "正常状态"),
    CWRUFileSpec("105", 1, "inner_race_007", "内圈故障，故障尺度 0.007 in"),
    CWRUFileSpec("118", 2, "ball_007", "滚动体故障，故障尺度 0.007 in"),
    CWRUFileSpec("130", 3, "outer_race_007", "外圈故障，故障尺度 0.007 in"),
    CWRUFileSpec("169", 4, "inner_race_014", "内圈故障，故障尺度 0.014 in"),
    CWRUFileSpec("185", 5, "ball_014", "滚动体故障，故障尺度 0.014 in"),
    CWRUFileSpec("197", 6, "outer_race_014", "外圈故障，故障尺度 0.014 in"),
    CWRUFileSpec("209", 7, "inner_race_021", "内圈故障，故障尺度 0.021 in"),
    CWRUFileSpec("222", 8, "ball_021", "滚动体故障，故障尺度 0.021 in"),
    CWRUFileSpec("234", 9, "outer_race_021", "外圈故障，故障尺度 0.021 in"),
)


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def expected_mat_path(data_root: Path, file_id: str) -> Path:
    return data_root / f"{file_id}.mat"


def aligned_csv_name() -> str:
    return "cwru_12k_drive_end_10class_aligned.csv"


def aligned_npz_name() -> str:
    return "cwru_12k_drive_end_10class_aligned.npz"


def windows_npz_name(window_size: int = DEFAULT_WINDOW_SIZE, step_size: int = DEFAULT_STEP_SIZE) -> str:
    return f"cwru_12k_drive_end_10class_windows_w{window_size}_s{step_size}.npz"


def label_names_array() -> np.ndarray:
    return np.array([spec.label_name for spec in FILE_SPECS])


def labels_array() -> np.ndarray:
    return np.array([spec.label_id for spec in FILE_SPECS], dtype=np.int64)


def find_drive_end_key(mat_data: dict[str, np.ndarray], file_id: str) -> str:
    expected_key = f"X{file_id}_DE_time"
    if expected_key in mat_data:
        return expected_key

    candidate_keys = sorted(key for key in mat_data.keys() if key.endswith("_DE_time"))
    if len(candidate_keys) == 1:
        return candidate_keys[0]
    raise KeyError(f"Cannot determine drive-end key for file {file_id}.")


def load_drive_end_signal(file_path: Path) -> tuple[np.ndarray, str, float | None]:
    from scipy.io import loadmat

    mat_data = loadmat(file_path)
    file_id = file_path.stem
    drive_end_key = find_drive_end_key(mat_data, file_id)
    signal = np.asarray(mat_data[drive_end_key]).reshape(-1).astype(np.float32)

    rpm_key = f"X{file_id}RPM"
    rpm = None
    if rpm_key in mat_data:
        rpm = float(np.asarray(mat_data[rpm_key]).reshape(-1)[0])

    return signal, drive_end_key, rpm


def truncate_signal(signal: np.ndarray, length: int = ALIGNED_LENGTH) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if len(signal) < length:
        raise ValueError(f"signal length {len(signal)} is shorter than required length {length}")
    return signal[:length].astype(np.float32, copy=False)


def slice_signal(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive integers")
    if len(signal) < window_size:
        raise ValueError("signal is shorter than the window size")

    windows = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])
    return np.stack(windows, axis=0).astype(np.float32, copy=False)


def save_npz(npz_path: Path, signals: np.ndarray, labels: np.ndarray, label_names: np.ndarray) -> None:
    np.savez_compressed(
        npz_path,
        signals=signals.astype(np.float32, copy=False),
        labels=labels.astype(np.int64, copy=False),
        label_names=label_names,
    )
