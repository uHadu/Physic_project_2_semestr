"""Проверка, что схемы правильно собраны на ядре (presets.py) и серия снимается верно."""
import numpy as np
import pytest

import presets
from experiment import emf_grid, run_series
from fitting import analyze

R_A, R_V = 0.5, 1e6
# Почти идеальная шкала: без шума и без заметного округления показаний
EXACT = dict(R_A=R_A, R_V=R_V, counts=10 ** 9)
R_LIST = [1.0, 100.0, 1e4, 1e5, 1e6]


def readings(circuit):
    rep = circuit.solve().reports(noise=False)
    return rep[presets.VOLTMETER]["reading"], rep[presets.AMMETER]["reading"]


@pytest.mark.parametrize("R", R_LIST)
def test_scheme_a_measures_R_parallel_RV(R):
    U, I = readings(presets.scheme_a(R=R, **EXACT))
    assert U / I == pytest.approx(R * R_V / (R + R_V), rel=1e-5)


@pytest.mark.parametrize("R", R_LIST)
def test_scheme_b_measures_R_plus_RA(R):
    U, I = readings(presets.scheme_b(R=R, **EXACT))
    assert U / I == pytest.approx(R + R_A, rel=1e-5)


def test_readme_check_scheme_a_underestimates_by_9_1_percent():
    """README: при R = 100 кОм и R_V = 1 МОм схема А занижает результат ≈ на 9,1 %."""
    s = run_series(presets.scheme_a(R=1e5, **EXACT), emf_grid(10, 10), noise=False)
    a = analyze(s.U, s.I, R_set=1e5)
    assert a.R < 1e5
    assert a.delta == pytest.approx(100 / 11, abs=0.01)     # 9,09 %


def test_series_and_parallel_presets():
    ideal = dict(R_A=0.0, R_V=float("inf"), counts=10 ** 9)
    U, I = readings(presets.series_resistors(R1=1e3, R2=2.2e3, **ideal))
    assert U / I == pytest.approx(3.2e3, rel=1e-6)
    U, I = readings(presets.parallel_resistors(R1=1e3, R2=1e3, **ideal))
    assert U / I == pytest.approx(500.0, rel=1e-6)


def test_switch_is_right_after_source_and_open_switch_breaks_circuit():
    c = presets.scheme_a(R=100.0, closed=False)
    assert c[presets.SWITCH].nodes == [presets.NODE_SRC, presets.NODE_K]
    assert c[presets.AMMETER].nodes[0] == presets.NODE_K
    U, I = readings(c)
    assert abs(I) < 1e-9


def test_lamp_is_nonlinear_and_burns_and_new_circuit_is_fresh():
    c = presets.lamp_circuit("6V_3W", R_A=0.0, R_V=float("inf"), counts=10 ** 9)
    s = run_series(c, np.linspace(0.5, 6.0, 12), noise=False)
    R_static = s.U / s.I
    assert R_static[-1] > 3 * R_static[0]            # нить нагрелась — R выросло
    assert not analyze(s.U, s.I).is_linear

    s = run_series(c, [5.0, 12.0], noise=False)      # перегрузка по мощности
    assert s.burned[-1] and c["L"].burned
    assert not s.valid[-1]
    assert any("перегорела" in w for w in s.warnings)
    # перегоревшая лампа остаётся перегоревшей — для нового эксперимента собираем цепь заново
    assert not presets.lamp_circuit("6V_3W")["L"].burned


def test_noise_is_reproducible_and_differs_between_points():
    c = presets.scheme_a(R=1e5)
    U1, I1 = run_series(c, [5.0] * 6, seed=7)
    U2, I2 = run_series(presets.scheme_a(R=1e5), [5.0] * 6, seed=7)
    assert np.array_equal(I1, I2)                    # тот же seed — те же показания
    assert len(set(I1)) > 1                          # но шум в точках разный


def test_meter_overload_marks_point():
    c = presets.scheme_b(R=1.0, R_A=0.0)             # ток > 10 А — предел амперметра
    s = run_series(c, [1.0, 20.0], noise=False)
    assert np.isnan(s.I[-1]) and not s.valid[-1]
    assert any("Перегрузка" in w for w in s.warnings)


def test_build_registry():
    assert set(presets.PRESETS) >= {"A", "B", "series", "parallel", "lamp"}
    c = presets.build("B", R=100.0, R1=1.0, lamp_model="x")   # лишние параметры отбрасываются
    assert c[presets.VOLTMETER].nodes[0] == presets.NODE_K
    with pytest.raises(ValueError):
        presets.build("Z")
