from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingParameters:
    """滚动轴承几何参数。

    Parameters
    ----------
    N : int
        滚动体个数。
    d : float
        滚动体直径（mm）。
    D : float
        轴承节圆直径（mm），即内外圈沟槽圆心的平均直径。
    alpha : float
        接触角（度）。深沟球轴承通常取 0°。
    """

    N: int
    d: float
    D: float
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be a positive integer")
        if self.d <= 0 or self.D <= 0:
            raise ValueError("d and D must be positive")
        if self.D <= self.d:
            raise ValueError("D (pitch diameter) must be greater than d (ball diameter)")


class BearingCharacteristicFrequencyCalculator:
    """根据轴承几何参数和转速计算四类特征频率。

    特征频率公式（接触角 alpha，单位：度）：

    设 f_r = rpm / 60，beta = d/D * cos(alpha)

    BPFO = (N/2) * f_r * (1 - beta)
    BPFI = (N/2) * f_r * (1 + beta)
    BSF  = (D / (2*d)) * f_r * (1 - beta^2)
    CF   = (f_r / 2) * (1 - beta)
    """

    def __init__(self, params: BearingParameters) -> None:
        self.params = params
        self._cos_alpha = math.cos(math.radians(params.alpha))
        self._beta = (params.d / params.D) * self._cos_alpha

    def _fr(self, rpm: float) -> float:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        return rpm / 60.0

    def bpfo(self, rpm: float) -> float:
        """外圈故障频率 BPFO（Hz）。"""
        return (self.params.N / 2.0) * self._fr(rpm) * (1.0 - self._beta)

    def bpfi(self, rpm: float) -> float:
        """内圈故障频率 BPFI（Hz）。"""
        return (self.params.N / 2.0) * self._fr(rpm) * (1.0 + self._beta)

    def bsf(self, rpm: float) -> float:
        """滚动体故障频率 BSF（Hz）。"""
        return (self.params.D / (2.0 * self.params.d)) * self._fr(rpm) * (1.0 - self._beta ** 2)

    def cf(self, rpm: float) -> float:
        """保持架频率 CF（Hz）。"""
        return (self._fr(rpm) / 2.0) * (1.0 - self._beta)

    def calculate_all_frequencies(self, rpm: float) -> dict[str, float]:
        """返回四类特征频率的字典，键为 'BPFO'、'BPFI'、'BSF'、'CF'。"""
        return {
            "BPFO": self.bpfo(rpm),
            "BPFI": self.bpfi(rpm),
            "BSF":  self.bsf(rpm),
            "CF":   self.cf(rpm),
        }


@dataclass(frozen=True)
class GearboxParameters:
    """齿轮箱基本参数。

    Parameters
    ----------
    pinion_teeth : int
        驱动齿轮（小齿轮）齿数。
    gear_teeth : int
        从动齿轮齿数。
    pinion_rpm : float
        驱动齿轮转速（rpm）。
    """

    pinion_teeth: int
    gear_teeth: int
    pinion_rpm: float

    def __post_init__(self) -> None:
        if self.pinion_teeth <= 0 or self.gear_teeth <= 0:
            raise ValueError("Tooth counts must be positive integers")
        if self.pinion_rpm <= 0:
            raise ValueError("pinion_rpm must be positive")


class GearboxCharacteristicFrequencyCalculator:
    """计算齿轮箱的啮合频率与边带频率。"""

    def __init__(self, pinion_teeth: int, gear_teeth: int, pinion_rpm: float) -> None:
        self.params = GearboxParameters(
            pinion_teeth=pinion_teeth,
            gear_teeth=gear_teeth,
            pinion_rpm=pinion_rpm,
        )
        self._pinion_fr = pinion_rpm / 60.0
        self._gear_fr = self._pinion_fr * pinion_teeth / gear_teeth

    @property
    def pinion_fr(self) -> float:
        """驱动齿轮旋转频率（Hz）。"""
        return self._pinion_fr

    @property
    def gear_fr(self) -> float:
        """从动齿轮旋转频率（Hz）。"""
        return self._gear_fr

    def calculate_gmf(self) -> float:
        """齿轮啮合频率 GMF（Hz）。"""
        return self.params.pinion_teeth * self._pinion_fr

    def get_sidebands(self, num_sidebands: int = 3) -> dict[str, list[float]]:
        """返回以驱动齿轮频率为调制源的 GMF 边带频率。

        Parameters
        ----------
        num_sidebands : int
            单侧边带阶数（默认 3 阶）。

        Returns
        -------
        dict with keys 'lower', 'gmf', 'upper'
        """
        gmf = self.calculate_gmf()
        lower = [round(gmf - k * self._pinion_fr, 4) for k in range(1, num_sidebands + 1)]
        upper = [round(gmf + k * self._pinion_fr, 4) for k in range(1, num_sidebands + 1)]
        return {
            "lower": lower[::-1],  # 从远到近排列
            "gmf": round(gmf, 4),
            "upper": upper,
        }

    def gear_ratio(self) -> float:
        """传动比（从动 / 驱动）。"""
        return self.params.gear_teeth / self.params.pinion_teeth
