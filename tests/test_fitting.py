"""Тесты обработки (fitting.py): МНК, ΔR, проверка линейности, δ."""
import math

import numpy as np
import pytest

from fitting import analyze, fit_linear, fit_through_origin, relative_error, student_t


@pytest.mark.parametrize("R", [1.0, 100.0, 1e4, 1e5, 4.7e6])
def test_ideal_data_gives_exact_R(R):
    """На идеальных данных I = U/R МНК даёт R_эксп = R_задан, ΔR = 0, δ = 0."""
    U = np.arange(1.0, 11.0)
    a = analyze(U, U / R, R_set=R)
    assert a.R == pytest.approx(R, rel=1e-12)
    assert a.dR == pytest.approx(0.0, abs=1e-9 * R)
    assert a.delta == pytest.approx(0.0, abs=1e-9)
    assert a.linear.r2 == pytest.approx(1.0)
    assert a.linear.b == pytest.approx(0.0, abs=1e-12 / R)
    assert a.is_linear


def test_student_coefficient_from_readme():
    """README: t ≈ 2,26 для n = 10 точек (n − 1 = 9), P = 0,95."""
    assert student_t(9) == pytest.approx(2.262, abs=1e-3)


def test_formulas_match_readme_by_hand():
    U = np.array([1.0, 2.0, 3.0, 4.0])
    I = np.array([0.011, 0.019, 0.031, 0.040])
    f = fit_through_origin(U, I)
    k = np.sum(U * I) / np.sum(U * U)
    S_k = math.sqrt(np.sum((I - k * U) ** 2) / ((len(U) - 1) * np.sum(U * U)))
    assert f.k == pytest.approx(k)
    assert f.S_k == pytest.approx(S_k)
    assert f.R == pytest.approx(1 / k)
    assert f.dR == pytest.approx(S_k / k ** 2 * student_t(3))


def test_noisy_data_within_interval_most_of_the_time():
    """С шумом R_задан попадает в R_эксп ± ΔR примерно в 95 % серий."""
    rng = np.random.default_rng(1)
    U = np.arange(1.0, 11.0)
    R = 100.0
    hits = 0
    for _ in range(400):
        I = U / R + rng.normal(0, 2e-4, U.size)
        f = fit_through_origin(U, I)
        hits += abs(f.R - R) <= f.dR
    assert 0.9 < hits / 400 <= 1.0


def test_linear_fit_detects_offset():
    """ВАХ со сдвигом (I = U/R + b) не проходит проверку линейности через ноль."""
    U = np.arange(1.0, 11.0)
    I = U / 100 + 0.02 + np.array([1, -1] * 5) * 1e-4
    lin = fit_linear(U, I)
    assert lin.b == pytest.approx(0.02, abs=1e-3)
    assert not lin.b_is_zero


def test_relative_error():
    assert relative_error(90.909, 100.0) == pytest.approx(9.091)
    assert relative_error(110.0, 100.0) == pytest.approx(10.0)


def test_nan_points_are_ignored_and_too_few_points_rejected():
    U = np.array([1.0, 2.0, np.nan, 4.0])
    f = fit_through_origin(U, U / 50)
    assert f.n == 3 and f.R == pytest.approx(50)
    with pytest.raises(ValueError):
        fit_through_origin([1.0], [0.1])
