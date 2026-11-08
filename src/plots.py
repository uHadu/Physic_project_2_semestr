"""
plots.py — графики и таблица результатов (раздел 5 README).

  plot_iv          — ВАХ I(U): точки с крестами погрешностей, аппроксимирующая прямая,
                     подпись R_эксп ± ΔR;
  plot_iv_grid     — все эксперименты на одном рисунке;
  results_table    — таблица R_задан | схема | R_эксп ± ΔR | δ (+ служебные столбцы);
  save_table_csv   — сохранение таблицы в CSV (разделитель «;», десятичная запятая);
  plot_delta_vs_R  — ошибка метода δ(R) для схем А и Б, логарифмические шкалы.

Все подписи на русском, числа на подписях — с десятичной запятой.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Цвета серий: схема А — синий, Б — оранжевый (плюс разные маркеры — не только цвет)
COLORS = {"A": "#2a78d6", "B": "#eb6834"}
MARKERS = {"A": "o", "B": "s"}
DEFAULT_COLOR = "#2a78d6"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"

SCHEME_RU = {"A": "А", "B": "Б"}

_PREFIXES = [(1e6, "М"), (1e3, "к"), (1.0, ""), (1e-3, "м"), (1e-6, "мк"), (1e-9, "н")]


# ---------------------------------------------------------------------------
# Форматирование чисел: десятичная запятая, приставки
# ---------------------------------------------------------------------------

def comma(x: float, fmt: str = "g") -> str:
    """Число с десятичной запятой: comma(2.5) → '2,5'."""
    return format(x, fmt).replace(".", ",").replace("-", "−")


def _prefix(value: float, unit: str) -> tuple[float, str]:
    """Подобрать приставку. Для «Ом» дольные приставки не используем (0,5 Ом)."""
    a = abs(value)
    for factor, p in _PREFIXES:
        if a >= factor or (unit == "Ом" and factor == 1.0) or factor == _PREFIXES[-1][0]:
            return factor, p
    return 1.0, ""


def fmt_pm(value: float, err: float, unit: str = "Ом") -> str:
    """Значение ± погрешность: погрешность округляется до 2 значащих цифр,
    значение — до того же разряда. fmt_pm(90912.3, 812.4) → '(90,91 ± 0,81) кОм'."""
    if not math.isfinite(value):
        return "—"
    factor, p = _prefix(value, unit)
    v, e = value / factor, err / factor
    if not math.isfinite(e):
        return f"{comma(v, '.4g')} ± ∞ {p}{unit}"
    if e <= 0:
        return f"{comma(v, '.6g')} ± 0 {p}{unit}"
    decimals = max(0, 1 - int(math.floor(math.log10(e))))
    return f"({comma(v, f'.{decimals}f')} ± {comma(e, f'.{decimals}f')}) {p}{unit}"


def fmt_value(value: float, unit: str, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    factor, p = _prefix(value, unit)
    return f"{comma(value / factor, f'.{digits}g')} {p}{unit}"


def _axis_scale(values, unit: str) -> tuple[float, str]:
    m = np.nanmax(np.abs(values)) if np.size(values) and np.isfinite(values).any() else 1.0
    factor, p = _prefix(m if m > 0 else 1.0, unit)
    return factor, f"{p}{unit}"


def _comma_ticks(ax) -> None:
    f = FuncFormatter(lambda v, _: comma(round(v, 10), "g"))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


# ---------------------------------------------------------------------------
# ВАХ
# ---------------------------------------------------------------------------

def plot_iv(series, analysis, R_set: float | None = None, title: str | None = None,
            ax=None, scheme: str = "A"):
    """График I(U): точки с крестами погрешностей (±Δ приборов из ядра) и прямая МНК I = U/R_эксп."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure
    color = COLORS.get(scheme, DEFAULT_COLOR)
    ok = series.valid
    fu, uu = _axis_scale(series.U[ok] if ok.any() else series.U, "В")
    fi, ui = _axis_scale(series.I[ok] if ok.any() else series.I, "А")

    ax.errorbar(series.U[ok] / fu, series.I[ok] / fi,
                xerr=series.dU[ok] / fu, yerr=series.dI[ok] / fi,
                fmt=MARKERS.get(scheme, "o"), ms=6, color=color, ecolor=color,
                elinewidth=1.2, capsize=3, zorder=3, label="измерения (±Δ приборов)")
    bad = ~ok & np.isfinite(series.U) & np.isfinite(series.I)
    if bad.any():
        ax.plot(series.U[bad] / fu, series.I[bad] / fi, "x", color=MUTED, ms=7,
                label="не обрабатываются (лампа перегорела)")

    U_max = float(np.nanmax(series.U[ok])) if ok.any() else 1.0
    u = np.linspace(0.0, 1.05 * U_max, 50)
    k = analysis.origin.k
    ax.plot(u / fu, k * u / fi, color=INK, lw=2.0, zorder=2,
            label=f"МНК: I = U / R_эксп,  R_эксп = {fmt_pm(analysis.R, analysis.dR)}")
    if R_set:
        ax.plot(u / fu, u / R_set / fi, "--", color=MUTED, lw=1.4, zorder=1,
                label=f"закон Ома I = U / R_задан,  R_задан = {fmt_value(R_set, 'Ом')}")

    lines = []
    if analysis.linear is not None:
        lin = analysis.linear
        b_txt = fmt_value(lin.b, "А", 3) if lin.b != 0 else "0 А"
        db_txt = fmt_value(lin.db, "А", 2) if lin.db > 0 else "0 А"
        lines.append(f"I = kU + b:  b = {b_txt} ± {db_txt}")
        lines.append(f"R² = {comma(lin.r2, '.6f') if math.isfinite(lin.r2) else '—'}")
        lines.append("ВАХ линейна: b ≈ 0 в пределах погрешности" if analysis.is_linear
                     else "ВАХ НЕ линейна (b ≠ 0 или R² < 0,99)")
    if analysis.delta is not None:
        lines.append(f"δ = {comma(analysis.delta, '.2f')} %")
    if lines:
        ax.text(0.98, 0.04, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    ax.set_xlabel(f"U, {uu}", color=INK)
    ax.set_ylabel(f"I, {ui}", color=INK)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=11, color=INK)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)
    _comma_ticks(ax)
    return fig


def plot_iv_grid(rows):
    """Все эксперименты: строки — R, столбцы — схемы."""
    R_vals = sorted({r["R_set"] for r in rows})
    schemes = sorted({r["scheme"] for r in rows})
    fig, axes = plt.subplots(len(R_vals), len(schemes), figsize=(6.4 * len(schemes), 4.0 * len(R_vals)),
                             squeeze=False)
    for row in rows:
        ax = axes[R_vals.index(row["R_set"])][schemes.index(row["scheme"])]
        plot_iv(row["series"], row["fit"], R_set=row["R_set"], ax=ax, scheme=row["scheme"],
                title=f"Схема {SCHEME_RU.get(row['scheme'], row['scheme'])}, "
                      f"R_задан = {fmt_value(row['R_set'], 'Ом', 3)}")
    fig.suptitle("ВАХ резистора в схемах А и Б", fontsize=14, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Таблица
# ---------------------------------------------------------------------------

TABLE_COLUMNS = ["R_задан, Ом", "Схема", "R_эксп ± ΔR, Ом", "δ, %"]


def results_table(rows) -> pd.DataFrame:
    """Таблица раздела 5 README. Первые 4 столбца — как в README, дальше служебные:
    ошибка метода (по тем же точкам без шума), проверка линейности, число точек."""
    recs = []
    for r in rows:
        fit = r["fit"]
        exact = r.get("exact")
        R_set = r.get("R_set")
        lin = fit.linear
        recs.append({
            "R_задан, Ом": fmt_value(R_set, "Ом") if R_set else "—",
            "Схема": SCHEME_RU.get(r["scheme"], r["scheme"]),
            "R_эксп ± ΔR, Ом": fmt_pm(fit.R, fit.dR),
            "δ, %": comma(fit.delta, ".2f") if fit.delta is not None else "—",
            "δ метода, %": (comma(exact.delta, ".3f")
                            if exact is not None and exact.delta is not None else "—"),
            "b ± Δb": (f"{fmt_value(lin.b, 'А', 3)} ± {fmt_value(lin.db, 'А', 2)}"
                       if lin is not None else "—"),
            "R²": comma(lin.r2, ".5f") if lin is not None and math.isfinite(lin.r2) else "—",
            "ВАХ линейна": {True: "да", False: "нет", None: "—"}[fit.is_linear],
            "n": fit.origin.n,
        })
    return pd.DataFrame(recs)


def save_table_csv(table: pd.DataFrame, path: str) -> None:
    """CSV для Excel/LibreOffice в русской локали: разделитель «;», UTF-8 с BOM."""
    table.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Ошибка метода δ(R)
# ---------------------------------------------------------------------------

def plot_delta_vs_R(R_grid, curves: dict, R_A: float, R_V: float, rows=None, ax=None):
    """δ метода от R для схем А и Б (лог. шкалы) + отмечен порог √(R_A·R_V).

    curves — {"A": δ(R) в %, "B": ...}, rows — результаты эксперимента с шумом (точки).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
    else:
        fig = ax.figure
    floor = 1e-7
    for s, d in curves.items():
        ax.plot(R_grid, np.maximum(d, floor), color=COLORS.get(s, DEFAULT_COLOR), lw=2.0,
                label=f"схема {SCHEME_RU.get(s, s)}: ошибка метода (без шума)")
    if rows:
        for s in curves:
            pts = [(r["R_set"], r["fit"].delta) for r in rows if r["scheme"] == s]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, np.maximum(y, floor), MARKERS.get(s, "o"), ms=8, mfc="white", mew=2,
                        color=COLORS.get(s, DEFAULT_COLOR),
                        label=f"схема {SCHEME_RU.get(s, s)}: эксперимент с шумом")
    R_cross = math.sqrt(R_A * R_V)
    ax.axvline(R_cross, color=MUTED, ls="--", lw=1.2)
    ax.text(R_cross * 1.15, 0.97, f"√(R_A·R_V) = {fmt_value(R_cross, 'Ом', 3)}",
            transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R_задан, Ом", color=INK)
    ax.set_ylabel("δ, %", color=INK)
    ax.set_title(f"Систематическая ошибка метода: R_A = {fmt_value(R_A, 'Ом', 3)}, "
                 f"R_V = {fmt_value(R_V, 'Ом', 3)}", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower center", ncol=2)
    _style(ax)
    log_fmt = FuncFormatter(lambda v, _: comma(v, "g"))
    ax.xaxis.set_major_formatter(log_fmt)
    ax.yaxis.set_major_formatter(log_fmt)
    return fig
