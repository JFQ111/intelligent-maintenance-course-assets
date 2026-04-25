"""25 个人工诊断特征的计算工具。

输入:
    signal: 1D 振动信号, shape [L]
    fs: 采样频率

输出:
    extract_25_features -> 1D 特征向量, shape [25]
    extract_feature_matrix -> 2D 特征矩阵, shape [N, 25]
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


EPS = 1e-12

FEATURE_NAMES = [
    "均值",
    "绝对均值",
    "均方根值",
    "方差",
    "标准差",
    "峰值",
    "峰峰值",
    "偏度",
    "峭度",
    "波形因子",
    "峰值因子",
    "脉冲因子",
    "裕度因子",
    "重心频率",
    "均方频率",
    "均方根频率",
    "频率方差",
    "频率标准差",
    "主频位置",
    "主频幅值",
    "谱均值",
    "谱方差",
    "谱偏度",
    "谱峭度",
    "谱熵",
]


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


def _safe_skew(values: np.ndarray) -> float:
    result = skew(values, bias=False)
    return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))


def _safe_kurtosis(values: np.ndarray) -> float:
    result = kurtosis(values, fisher=False, bias=False)
    return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))


def extract_25_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """提取单个样本的 25 维特征。

    参数:
        signal: shape [L]
        fs: 采样频率

    返回:
        features: shape [25], dtype float32
    """

    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("Input signal is empty.")

    mean_value = float(np.mean(x))
    abs_mean = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    variance = float(np.var(x))
    std_value = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.max(x) - np.min(x))
    skewness = _safe_skew(x)
    kurt_value = _safe_kurtosis(x)
    waveform_factor = _safe_divide(rms, abs_mean)
    crest_factor = _safe_divide(peak, rms)
    impulse_factor = _safe_divide(peak, abs_mean)
    root_amplitude = float(np.mean(np.sqrt(np.abs(x))))
    clearance_factor = _safe_divide(peak, root_amplitude**2)

    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))

    if freqs.size > 1:
        freqs = freqs[1:]
        spectrum = spectrum[1:]
    else:
        freqs = np.zeros((1,), dtype=np.float64)
        spectrum = np.zeros((1,), dtype=np.float64)

    power = np.square(spectrum)
    power_sum = float(np.sum(power))

    if power_sum <= EPS:
        centroid_freq = 0.0
        mean_square_freq = 0.0
        root_mean_square_freq = 0.0
        freq_variance = 0.0
        freq_std = 0.0
        dominant_freq = 0.0
        dominant_amplitude = 0.0
        spectrum_mean = float(np.mean(spectrum))
        spectrum_variance = float(np.var(spectrum))
        spectrum_skewness = 0.0
        spectrum_kurtosis = 0.0
        spectrum_entropy = 0.0
    else:
        centroid_freq = float(np.sum(freqs * power) / power_sum)
        mean_square_freq = float(np.sum(np.square(freqs) * power) / power_sum)
        root_mean_square_freq = float(np.sqrt(mean_square_freq))
        freq_variance = float(np.sum(np.square(freqs - centroid_freq) * power) / power_sum)
        freq_std = float(np.sqrt(freq_variance))
        dominant_index = int(np.argmax(spectrum))
        dominant_freq = float(freqs[dominant_index])
        dominant_amplitude = float(spectrum[dominant_index])
        spectrum_mean = float(np.mean(spectrum))
        spectrum_variance = float(np.var(spectrum))
        spectrum_skewness = _safe_skew(spectrum)
        spectrum_kurtosis = _safe_kurtosis(spectrum)
        probability = power / (power_sum + EPS)
        spectrum_entropy = float(-np.sum(probability * np.log(probability + EPS)))

    features = np.array(
        [
            mean_value,
            abs_mean,
            rms,
            variance,
            std_value,
            peak,
            peak_to_peak,
            skewness,
            kurt_value,
            waveform_factor,
            crest_factor,
            impulse_factor,
            clearance_factor,
            centroid_freq,
            mean_square_freq,
            root_mean_square_freq,
            freq_variance,
            freq_std,
            dominant_freq,
            dominant_amplitude,
            spectrum_mean,
            spectrum_variance,
            spectrum_skewness,
            spectrum_kurtosis,
            spectrum_entropy,
        ],
        dtype=np.float32,
    )
    return features


def extract_feature_matrix(samples: np.ndarray, fs: float) -> np.ndarray:
    """提取多个样本的 25 维特征矩阵。

    参数:
        samples: shape [N, L]
        fs: 采样频率

    返回:
        X_features: shape [N, 25], dtype float32
    """

    samples_array = np.asarray(samples, dtype=np.float32)
    if samples_array.ndim != 2:
        raise ValueError(f"Expected samples with shape [N, L], got {samples_array.shape}.")
    feature_list = [extract_25_features(sample, fs) for sample in samples_array]
    return np.stack(feature_list, axis=0).astype(np.float32)
