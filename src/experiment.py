"""
experiment.py — серия измерений на ядре и вычислительный эксперимент задания 7.1.

run_series(circuit, emf_values, seed) — снять ВАХ: в цикле меняем ЭДС источника,
решаем цепь ядром (solve с x0 = предыдущее решение) и берём показания приборов
из sol.reports(noise=True).

Запуск как скрипт:
    python src/experiment.py
прогоняет R = 1 Ом, 100 Ом, 10 кОм, 100 кОм в схемах А и Б и сохраняет
таблицу и графики в results/.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np

from core_api import ROOT, Circuit, Lamp, Resistor
from presets import AMMETER, SOURCE, VOLTMETER

RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Результат серии
# ---------------------------------------------------------------------------

@dataclass
class Series:
    """Серия измерений. Можно распаковать как пару массивов: U, I = run_series(...).

    U, I     — показания вольтметра и амперметра (В, А); NaN — прибор перегружен;
    dU, dI   — предельные погрешности приборов из ядра (поле «error»: Δ = γ·предел/100 %);
    U_true, I_true — истинные U и I на приборах из решения ядра, без шума и округления
               (нужны, чтобы отделить ошибку метода от случайной);
    burned   — перегорела ли лампа к этой точке;
    warnings — предупреждения ядра (sol.warnings) и замечания по ходу серии;
    last_reports — отчёт ядра по всем элементам в последней точке.
    """
    emf: np.ndarray
    U: np.ndarray
    I: np.ndarray
    dU: np.ndarray
    dI: np.ndarray
    U_true: np.ndarray
    I_true: np.ndarray
    burned: np.ndarray
    warnings: list[str] = field(default_factory=list)
    last_reports: dict = field(default_factory=dict)

    def __iter__(self):
        yield self.U
        yield self.I

    def __len__(self) -> int:
        return len(self.emf)

    @property
    def valid(self) -> np.ndarray:
        """Точки, пригодные для обработки: нет перегрузки и лампа цела."""
        return np.isfinite(self.U) & np.isfinite(self.I) & ~self.burned

    def readings(self, i: int = -1) -> dict:
        """Показания i-й точки в формате circuit_view.draw_circuit(readings=...)."""
        return {"U_meas": _nan_to_none(self.U[i]), "I_meas": _nan_to_none(self.I[i])}


def comma_g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _nan_to_none(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def readings_from_reports(reports: dict, ammeter: str = AMMETER,
                          voltmeter: str = VOLTMETER) -> dict:
    """Показания приборов из sol.reports(...) → словарь для circuit_view (None — перегрузка)."""
    return {"U_meas": reports[voltmeter]["reading"], "I_meas": reports[ammeter]["reading"]}


# ---------------------------------------------------------------------------
# Серия измерений
# ---------------------------------------------------------------------------

def run_series(circuit: Circuit, emf_values, seed: int | None = None, noise: bool = True,
               auto_burn: bool = True, source: str = SOURCE, ammeter: str = AMMETER,
               voltmeter: str = VOLTMETER) -> Series:
    """Снять серию показаний при разных ЭДС источника.

    Цепь изменяется на месте: меняется ЭДС, а лампа может перегореть (lamp.burned
    остаётся True). Поэтому для каждого нового эксперимента цепь собирают заново
    (presets.build(...)).

    Шум: ядро в sol.reports(seed=...) каждый раз создаёт новый генератор random.Random(seed).
    Если отдать во все точки один и тот же seed, ошибка в каждой точке будет одинаковой,
    то есть станет систематической. Поэтому из seed серии выводится свой seed для каждой точки.
    Ошибки ядра (CircuitError, ElementError) пробрасываются наверх.
    """
    emf_values = np.asarray(list(emf_values), dtype=float)
    master = random.Random(seed)
    lamps = [el for el in circuit.elements.values() if isinstance(el, Lamp)]
    resistors = [el for el in circuit.elements.values() if isinstance(el, Resistor)]

    n = len(emf_values)
    U, I, dU, dI, U_true, I_true = (np.full(n, np.nan) for _ in range(6))
    burned = np.zeros(n, dtype=bool)
    warnings: list[str] = []

    def warn(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)

    overheated: set[str] = set()
    x0 = None
    reports: dict = {}
    for i, emf in enumerate(emf_values):
        circuit[source].set_param("emf", emf)
        sol = circuit.solve(x0=x0, auto_burn=auto_burn)
        x0 = sol.x
        for w in sol.warnings:
            warn(w)
        reports = sol.reports(noise=noise, seed=master.randrange(2 ** 32))
        a, v = reports[ammeter], reports[voltmeter]

        U_true[i], I_true[i] = v["U"], a["I"]
        if v["overload"] or a["overload"]:
            who = [n_ for n_, r in ((voltmeter, v), (ammeter, a)) if r["overload"]]
            warn(f"Перегрузка прибора {', '.join(who)} при ℰ = {comma_g(emf)} В — точка не обрабатывается")
        else:
            U[i], I[i] = v["reading"], a["reading"]
            dU[i], dI[i] = v["error"], a["error"]

        burned[i] = any(l.burned for l in lamps)
        if burned[i] and not burned[:i].any():
            warn(f"Лампа перегорела при ℰ = {comma_g(emf)} В (P > burn_factor·P_nom) — "
                 "дальше ток через неё не идёт, эти точки не обрабатываются")
        for r in resistors:
            if r.name not in overheated and reports[r.name].get("overheated"):
                overheated.add(r.name)
                warn((f"{r.name}: мощность {abs(reports[r.name]['P']):.3g} Вт больше допустимой "
                      f"P_max = {r.P_max:g} Вт уже при ℰ = {emf:g} В").replace(".", ",")
                     + " — реальный резистор нагреется (в модели ядра R от температуры не зависит)")

    return Series(emf_values, U, I, dU, dI, U_true, I_true, burned, warnings, reports)


def measure_once(circuit: Circuit, emf: float, seed: int | None = None,
                 noise: bool = True) -> tuple[dict, list[str]]:
    """Одно измерение: показания для circuit_view и предупреждения ядра."""
    s = run_series(circuit, [emf], seed=seed, noise=noise)
    return readings_from_reports(s.last_reports), s.warnings


def emf_grid(emf_max: float = 10.0, n: int = 10) -> np.ndarray:
    """ЭДС от ℰ_max/n до ℰ_max с равным шагом (при ℰ_max = 10 В, n = 10 — 1, 2, …, 10 В)."""
    if n < 2:
        raise ValueError("Нужно хотя бы 2 точки")
    return np.linspace(emf_max / n, emf_max, n)


# ---------------------------------------------------------------------------
# Вычислительный эксперимент (скрипт)
# ---------------------------------------------------------------------------

def run_experiment(R_values=(1.0, 100.0, 1e4, 1e5), schemes=("A", "B"), emf_max: float = 10.0,
                   n: int = 10, R_A: float = 0.5, R_V: float = 1e6, accuracy_class: float = 0.5,
                   seed: int = 2026) -> list[dict]:
    """Для каждого R и схемы: шумная серия + серия без шума (ошибка метода).

    Возвращает строки таблицы (словари) с сериями и результатами обработки.
    """
    import presets
    from fitting import analyze

    rows = []
    for j, R in enumerate(R_values):
        for scheme in schemes:
            params = dict(R=R, R_A=R_A, R_V=R_V, accuracy_class=accuracy_class)
            emfs = emf_grid(emf_max, n)
            # каждый эксперимент — новая цепь (лампы и ЭДС не тянутся из прошлого)
            series = run_series(presets.build(scheme, **params), emfs,
                                seed=seed + 100 * j + ord(scheme))
            fit = analyze(series.U[series.valid], series.I[series.valid], R_set=R)
            # те же точки по истинным U, I приборов: чистая ошибка метода
            exact = analyze(series.U_true, series.I_true, R_set=R)
            rows.append(dict(R_set=R, scheme=scheme, series=series, fit=fit, exact=exact))
    return rows


def method_error_curve(R_grid, scheme: str, R_A: float = 0.5, R_V: float = 1e6,
                       emf: float = 10.0) -> np.ndarray:
    """δ метода (%) для каждого R: одна точка без шума, R_изм = U/I по истинным величинам ядра."""
    import presets
    out = []
    for R in R_grid:
        s = run_series(presets.build(scheme, R=R, R_A=R_A, R_V=R_V), [emf], noise=False)
        R_meas = s.U_true[0] / s.I_true[0]
        out.append(abs(R_meas - R) / R * 100.0)
    return np.array(out)


def main(argv=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import circuit_view
    import plots

    ap = argparse.ArgumentParser(description="Вычислительный эксперимент: R по наклону ВАХ в схемах А и Б")
    ap.add_argument("--out", default=RESULTS_DIR, help="папка для результатов")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=10, help="число точек")
    ap.add_argument("--emf", type=float, default=10.0, help="максимальная ЭДС, В")
    ap.add_argument("--R_A", type=float, default=0.5)
    ap.add_argument("--R_V", type=float, default=1e6)
    ap.add_argument("--gamma", type=float, default=0.5, help="класс точности приборов, %%")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows = run_experiment(emf_max=args.emf, n=args.n, R_A=args.R_A, R_V=args.R_V,
                          accuracy_class=args.gamma, seed=args.seed)

    for row in rows:
        for w in row["series"].warnings:
            print(f"[схема {row['scheme']}, R = {row['R_set']:g} Ом] предупреждение: {w}")

    # таблица
    table = plots.results_table(rows)
    csv_path = os.path.join(args.out, "results_table.csv")
    plots.save_table_csv(table, csv_path)
    print(table.to_string(index=False))
    print("сохранено:", csv_path)

    # графики I(U) — по одному на эксперимент и сводный
    for row in rows:
        fig = plots.plot_iv(row["series"], row["fit"], R_set=row["R_set"],
                            title=f"Схема {plots.SCHEME_RU[row['scheme']]}, "
                                  f"R = {circuit_view.fmt(row['R_set'], 'Ом', 3)}")
        p = os.path.join(args.out, f"iv_{row['scheme']}_R{row['R_set']:g}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("сохранено:", p)
    fig = plots.plot_iv_grid(rows)
    p = os.path.join(args.out, "iv_all.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("сохранено:", p)

    # ошибка метода δ(R)
    # до 300 кОм: выше в схеме Б начинает сказываться утечка G_MIN = 1e-12 См
    # из каждого узла на землю в ядре (добавка −R·G_MIN к δ), см. docs/notes_for_partner.md
    R_grid = np.logspace(-1, math.log10(3e5), 61)
    curves = {s: method_error_curve(R_grid, s, args.R_A, args.R_V) for s in ("A", "B")}
    fig = plots.plot_delta_vs_R(R_grid, curves, R_A=args.R_A, R_V=args.R_V, rows=rows)
    p = os.path.join(args.out, "delta_vs_R.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("сохранено:", p)

    # схемы
    for p in circuit_view.save_figures(args.out, R=1e5, R_A=args.R_A, R_V=args.R_V):
        print("сохранено:", p)

    # контрольная проверка из README: R = 100 кОм, R_V = 1 МОм, схема А занижает ≈ на 9,1 %
    ref = next((r for r in rows if r["scheme"] == "A" and r["R_set"] == 1e5), None)
    if ref is not None:
        d = (ref["exact"].R - 1e5) / 1e5 * 100
        print(f"Проверка: схема А, R = 100 кОм, R_V = {args.R_V:g} Ом → ошибка метода "
              f"{d:+.2f} % (ожидается ≈ −9,1 % при R_V = 1 МОм); "
              f"с шумом δ = {ref['fit'].delta:.2f} %".replace(".", ","))


if __name__ == "__main__":
    main()
