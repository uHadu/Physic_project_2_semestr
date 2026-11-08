"""
app.py — интерфейс задания 7.1 «Закон Ома для участка цепи» на Streamlit.

Запуск из корня репозитория:
    streamlit run src/app.py

Схема собирается из элементов ядра (presets.py), серия измерений снимается
ядром (experiment.run_series), обработка — fitting.py, графики — plots.py,
рисунок схемы — circuit_view.py (показания на нём — из ядра).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import circuit_view  # noqa: E402
import plots  # noqa: E402
import presets  # noqa: E402
from core_api import CORE_ERRORS  # noqa: E402
from experiment import emf_grid, run_series  # noqa: E402
from fitting import analyze  # noqa: E402

# Значения для слайдеров (ряд E6 и декады)
R_VALUES = [round(m * 10 ** e, 6) for e in range(0, 7) for m in (1.0, 2.2, 4.7)] + [1e7]
R_A_VALUES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
R_V_VALUES = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, math.inf]
GAMMA_VALUES = [0.1, 0.2, 0.5, 1.0, 1.5, 2.5, 4.0]
EMF_VALUES = [k / 2 for k in range(1, 61)]          # 0,5 … 30 В с шагом 0,5 В


def ohm(v: float) -> str:
    return "∞ (идеальный)" if math.isinf(v) else circuit_view.fmt(v, "Ом", 3)


def comma(v: float) -> str:
    return f"{v:g}".replace(".", ",")


st.set_page_config(page_title="Закон Ома для участка цепи", layout="wide")
st.title("Задание 7.1. Закон Ома для участка цепи")
st.caption("Вольт-амперная характеристика резистора: R по наклону графика I(U) "
           "и систематическая ошибка схем включения приборов.")

# ---------------------------------------------------------------------------
# Управление
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Схема")
    key = st.selectbox("Готовая схема", list(presets.PRESETS),
                       format_func=lambda k: presets.PRESETS[k].title)
    preset = presets.PRESETS[key]
    params: dict = {}
    if preset.fixed_connection is None:
        params["connection"] = st.radio(
            "Вольтметр подключён", ["A", "B"], horizontal=True,
            format_func=lambda c: "к нагрузке (как в А)" if c == "A" else "к «A + нагрузка» (как в Б)")

    st.header("Нагрузка")
    if "R" in preset.load_params:
        params["R"] = st.select_slider("Сопротивление R", R_VALUES, value=1e5, format_func=ohm)
    if "R1" in preset.load_params:
        params["R1"] = st.select_slider("R₁", R_VALUES, value=1e3, format_func=ohm)
        params["R2"] = st.select_slider("R₂", R_VALUES, value=2.2e3, format_func=ohm)
    if "lamp_model" in preset.load_params:
        params["lamp_model"] = st.selectbox("Модель лампы (из ядра)", presets.LAMP_MODELS)

    st.header("Источник и серия")
    emf_max = st.select_slider("ЭДС ℰ (наибольшая в серии), В", EMF_VALUES, value=10.0, format_func=comma)
    n = st.slider("Число точек n", 3, 30, 10)
    st.caption(f"ℰ меняется от {comma(emf_max / n)} до {comma(emf_max)} В с шагом {comma(emf_max / n)} В")

    with st.expander("Приборы", expanded=True):
        params["R_A"] = st.select_slider("Сопротивление амперметра R_A", R_A_VALUES, value=0.5, format_func=ohm)
        params["R_V"] = st.select_slider("Сопротивление вольтметра R_V", R_V_VALUES, value=1e6, format_func=ohm)
        params["accuracy_class"] = st.select_slider("Класс точности γ, %", GAMMA_VALUES, value=0.5,
                                                    format_func=comma)
        noise = st.checkbox("Случайная погрешность приборов (шум)", value=True)
        seed = int(st.number_input("Зерно генератора (seed)", 0, 10 ** 6, 2026, step=1))

    run = st.button("Провести эксперимент", type="primary", width="stretch")

conn = preset.connection(params)
R_nominal = preset.nominal(params)

# ---------------------------------------------------------------------------
# Эксперимент
# ---------------------------------------------------------------------------
if "rows" not in st.session_state:
    st.session_state.rows = []

if run:
    try:
        circuit = preset.build(emf=emf_max, **params)   # новая цепь на каждый эксперимент
        series = run_series(circuit, emf_grid(emf_max, n), seed=seed, noise=noise)
        if series.valid.sum() < 2:
            raise ValueError("Меньше двух пригодных точек (перегрузка приборов или перегорела лампа) — "
                             "уменьшите ЭДС или выберите другие параметры")
        fit = analyze(series.U[series.valid], series.I[series.valid], R_set=R_nominal)
        exact = analyze(series.U_true[series.valid], series.I_true[series.valid], R_set=R_nominal)
    except CORE_ERRORS as exc:
        st.error(f"Ошибка ядра: {exc}")
    except ValueError as exc:
        st.error(f"Не удалось обработать серию: {exc}")
    else:
        label = conn if key in ("A", "B") else \
            f"{preset.title.split(' (')[0]}, вольтметр как в {plots.SCHEME_RU[conn]}"
        last = dict(key=key, scheme=label, conn=conn, R_set=R_nominal, params=dict(params),
                    emf_max=emf_max, series=series, fit=fit, exact=exact)
        st.session_state.last = last
        st.session_state.rows.append(last)

last = st.session_state.get("last")

# ---------------------------------------------------------------------------
# Схема и график
# ---------------------------------------------------------------------------
col_scheme, col_plot = st.columns([1, 1])

with col_scheme:
    st.subheader("Схема")
    show_readings = st.checkbox("Показать показания приборов", value=last is not None)
    readings = None
    view = dict(key=key, conn=conn, params=params, emf=emf_max)
    if last is not None:
        s = last["series"]
        idx = 0
        if len(s) > 1:
            idx = st.select_slider("Точка серии (ℰ, В)", list(range(len(s))), value=len(s) - 1,
                                   format_func=lambda i: comma(round(float(s.emf[i]), 6)))
        readings = s.readings(idx)
        view = dict(key=last["key"], conn=last["conn"], params=last["params"], emf=float(s.emf[idx]))

    p = view["params"]
    if view["key"] in ("A", "B"):
        load_label = None
    elif "lamp_model" in p:
        load_label = f"лампа {p['lamp_model']}"
    else:
        sign = " + " if view["key"] == "series" else " ∥ "
        load_label = f"R₁{sign}R₂ = {circuit_view.fmt(p['R1'], 'Ом', 3)}{sign}{circuit_view.fmt(p['R2'], 'Ом', 3)}"
    nominal = presets.PRESETS[view["key"]].nominal(p)

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    circuit_view.draw_circuit(
        ax, view["conn"], U_src=view["emf"], R=nominal or 1.0,
        R_A=p["R_A"], R_V=p["R_V"], readings=readings,
        show_readings=show_readings, load_label=load_label,
        show_formula=show_readings and nominal is not None and readings is not None)
    st.pyplot(fig)
    plt.close(fig)
    if last is not None and last["key"] != key:
        st.caption("Показана схема последнего эксперимента. Нажмите «Провести эксперимент» для новой.")

with col_plot:
    st.subheader("Вольт-амперная характеристика I(U)")
    if last is None:
        st.info("Выберите схему и параметры слева и нажмите «Провести эксперимент».")
    else:
        for w in last["series"].warnings:
            st.warning(w)
        fig = plots.plot_iv(last["series"], last["fit"], R_set=last["R_set"], scheme=last["conn"],
                            title=f"Схема {plots.SCHEME_RU[last['conn']]}" if last["key"] in ("A", "B")
                            else last["scheme"])
        st.pyplot(fig)
        plt.close(fig)
        fit = last["fit"]
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"**R_эксп ± ΔR**  \n### {plots.fmt_pm(fit.R, fit.dR)}")
        c2.metric("δ", f"{comma(round(fit.delta, 2))} %" if fit.delta is not None else "—")
        c3.metric("ВАХ линейна", {True: "да", False: "нет", None: "—"}[fit.is_linear])
        if not presets.PRESETS[last["key"]].linear:
            st.caption("Лампа — нелинейный элемент: сопротивление нити растёт с нагревом, "
                       "поэтому один наклон I(U) её не описывает (закон Ома для неё не выполняется).")

        with st.expander("Таблица измерений"):
            s = last["series"]
            st.dataframe(pd.DataFrame({
                "ℰ, В": s.emf, "U, В": s.U, "ΔU, В": s.dU, "I, А": s.I, "ΔI, А": s.dI,
                "обрабатывается": ["да" if v else "нет" for v in s.valid],
            }), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Сводная таблица
# ---------------------------------------------------------------------------
st.subheader("Результаты экспериментов")
if st.session_state.rows:
    table = plots.results_table(st.session_state.rows)
    st.dataframe(table, hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 1])
    c1.download_button("Скачать таблицу (CSV)", table.to_csv(sep=";", index=False).encode("utf-8-sig"),
                       file_name="results_table.csv", mime="text/csv")
    if c2.button("Очистить таблицу"):
        st.session_state.rows = []
        st.session_state.pop("last", None)
        st.rerun()
    st.caption("δ — отклонение R_эксп от R_задан с шумом; «δ метода» — по тем же точкам без шума "
               "(истинные U и I приборов из ядра), то есть чистая систематическая ошибка схемы.")
else:
    st.caption("Здесь будут накапливаться результаты: проведите эксперимент в схеме А и Б "
               "при одном R и сравните.")
