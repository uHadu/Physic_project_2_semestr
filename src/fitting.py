"""
fitting.py — обработка ВАХ по разделу 4 README.

1. МНК через начало координат:  k = ΣUI / ΣU²,  R_эксп = 1/k.
2. Погрешность наклона и сопротивления:
       S_k = √( Σ(I − kU)² / ((n − 1)·ΣU²) ),   ΔR = S_k / k² · t(P, n − 1),
   t — коэффициент Стьюдента для доверительной вероятности P = 0,95 (scipy.stats.t).
3. Проверка линейности: подгонка I = kU + b, b = 0 в пределах погрешности,
   коэффициент детерминации R².
4. Отклонение от заданного:  δ = |R_эксп − R_задан| / R_задан · 100 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.95


def student_t(n_dof: int, confidence: float = CONFIDENCE) -> float:
    """Коэффициент Стьюдента t для двустороннего интервала (n_dof степеней свободы).

    Для n = 10 точек (n − 1 = 9) и P = 0,95 это ≈ 2,262.
    """
    if n_dof < 1:
        raise ValueError("Для коэффициента Стьюдента нужна хотя бы одна степень свободы")
    return float(stats.t.ppf((1.0 + confidence) / 2.0, n_dof))


def _clean(U, I) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    I = np.asarray(I, dtype=float)
    if U.shape != I.shape:
        raise ValueError("Массивы U и I разной длины")
    ok = np.isfinite(U) & np.isfinite(I)
    return U[ok], I[ok]


@dataclass
class OriginFit:
    """Прямая I = kU через начало координат."""
    k: float          # наклон, См
    S_k: float        # СКО наклона
    R: float          # R_эксп = 1/k, Ом
    dR: float         # доверительная полуширина ΔR, Ом
    t: float          # коэффициент Стьюдента
    n: int


def fit_through_origin(U, I, confidence: float = CONFIDENCE) -> OriginFit:
    U, I = _clean(U, I)
    n = len(U)
    if n < 2:
        raise ValueError("Для МНК нужно хотя бы 2 точки")
    sU2 = float(np.sum(U * U))
    if sU2 == 0:
        raise ValueError("Все напряжения равны нулю — наклон не определён")
    k = float(np.sum(U * I)) / sU2
    resid = I - k * U
    S_k = math.sqrt(float(np.sum(resid ** 2)) / ((n - 1) * sU2))
    t = student_t(n - 1, confidence)
    R = 1.0 / k if k != 0 else math.inf
    dR = S_k / k ** 2 * t if k != 0 else math.inf
    return OriginFit(k, S_k, R, dR, t, n)


@dataclass
class LinearFit:
    """Прямая I = kU + b (проверка линейности)."""
    k: float
    b: float
    S_k: float
    S_b: float
    db: float          # доверительная полуширина для b (t для n − 2 степеней свободы)
    r2: float          # коэффициент детерминации R²
    b_is_zero: bool    # |b| ≤ Δb — свободный член равен нулю в пределах погрешности
    n: int


def fit_linear(U, I, confidence: float = CONFIDENCE) -> LinearFit:
    U, I = _clean(U, I)
    n = len(U)
    if n < 3:
        raise ValueError("Для подгонки I = kU + b нужно хотя бы 3 точки")
    Um = float(U.mean())
    Sxx = float(np.sum((U - Um) ** 2))
    if Sxx == 0:
        raise ValueError("Все напряжения одинаковые — наклон не определён")
    k = float(np.sum((U - Um) * (I - I.mean()))) / Sxx
    b = float(I.mean()) - k * Um
    resid = I - (k * U + b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((I - I.mean()) ** 2))
    s2 = ss_res / (n - 2)
    S_k = math.sqrt(s2 / Sxx)
    S_b = math.sqrt(s2 * float(np.sum(U * U)) / (n * Sxx))
    db = S_b * student_t(n - 2, confidence)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    # при идеальных данных db = 0, а b — машинный ноль: сравниваем с допуском
    tol = 1e-12 * max(abs(float(I.max())), abs(float(I.min())), 1e-300)
    return LinearFit(k, b, S_k, S_b, db, r2, abs(b) <= db + tol, n)


def relative_error(R_exp: float, R_set: float) -> float:
    """δ = |R_эксп − R_задан| / R_задан · 100 %."""
    return abs(R_exp - R_set) / R_set * 100.0


@dataclass
class Analysis:
    origin: OriginFit
    linear: LinearFit | None
    R_set: float | None
    delta: float | None        # δ, %

    @property
    def R(self) -> float:
        return self.origin.R

    @property
    def dR(self) -> float:
        return self.origin.dR

    @property
    def is_linear(self) -> bool | None:
        """Признак линейности ВАХ: b = 0 в пределах погрешности и R² близок к 1."""
        if self.linear is None:
            return None
        return self.linear.b_is_zero and self.linear.r2 > 0.99


def analyze(U, I, R_set: float | None = None, confidence: float = CONFIDENCE) -> Analysis:
    """Полная обработка серии: МНК через 0, проверка линейности, δ."""
    origin = fit_through_origin(U, I, confidence)
    try:
        linear = fit_linear(U, I, confidence)
    except ValueError:
        linear = None          # мало точек для проверки линейности
    delta = relative_error(origin.R, R_set) if R_set else None
    return Analysis(origin, linear, R_set, delta)
