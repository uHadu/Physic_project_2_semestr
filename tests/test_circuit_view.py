"""circuit_view рисует обе схемы без ошибок; показания — только из readings."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import circuit_view  # noqa: E402


def _texts(ax):
    return [t.get_text() for t in ax.texts]


@pytest.mark.parametrize("scheme", ["A", "B"])
@pytest.mark.parametrize("equivalent", [False, True])
def test_draws_without_readings(scheme, equivalent):
    fig, ax = plt.subplots()
    rd = circuit_view.draw_circuit(ax, scheme, U_src=10, R=1e5, equivalent=equivalent)
    assert rd == {}
    assert not any(t.startswith(("I =", "U =")) for t in _texts(ax))   # без show_readings — без дисплеев
    plt.close(fig)


@pytest.mark.parametrize("scheme", ["A", "B"])
def test_readings_come_from_core(scheme):
    readings = circuit_view.core_readings(scheme, U_src=10, R=1e5, noise=True, seed=3)
    fig, ax = plt.subplots()
    rd = circuit_view.draw_circuit(ax, scheme, U_src=10, R=1e5, readings=readings,
                                   show_readings=True, show_formula=True)
    texts = _texts(ax)
    assert "I = " + circuit_view.fmt(readings["I_meas"], "А") in texts
    assert "U = " + circuit_view.fmt(readings["U_meas"], "В") in texts
    assert rd["R_meas"] == pytest.approx(readings["U_meas"] / readings["I_meas"])
    plt.close(fig)


def test_show_readings_without_readings_shows_dash_and_handles_overload():
    fig, ax = plt.subplots()
    circuit_view.draw_circuit(ax, "A", show_readings=True, R_V=float("inf"))
    assert "I = —" in _texts(ax)
    circuit_view.draw_circuit(ax, "B", show_readings=True,
                              readings={"U_meas": 1.0, "I_meas": None}, load_label="лампа 6V_3W")
    assert "I = перегрузка" in _texts(ax) and "лампа 6V_3W" in _texts(ax)
    plt.close(fig)


def test_save_figures(tmp_path):
    paths = circuit_view.save_figures(str(tmp_path))
    assert paths and all((tmp_path / p.split("/")[-1]).stat().st_size > 0 for p in paths)
