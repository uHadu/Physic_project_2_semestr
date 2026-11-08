"""
circuit_view.py — отрисовка электрических схем для задания 7.1
«Закон Ома для участка цепи».

Рисует две схемы включения приборов (обозначения по ГОСТ 2.728 / 2.729):

  Схема А — вольтметр подключён непосредственно к резистору
            (амперметр меряет ток резистора + ток вольтметра);
  Схема Б — вольтметр подключён к цепочке «амперметр + резистор»
            (вольтметр меряет напряжение на резисторе + на амперметре).

Каждую схему можно нарисовать в двух видах:
  equivalent=False — обычная схема с приборами (кружки A и V);
  equivalent=True  — эквивалентная схема: приборы заменены их
                     внутренними сопротивлениями R_A и R_V (видно,
                     откуда берётся систематическая ошибка метода).

Показания приборов выводятся на схеме только при show_readings=True.
Сами показания схема НЕ считает: их передают параметром readings —
словарь {'U_meas': ..., 'I_meas': ...}, снятый с приборов ядра
(solver.py / components.py). Получить его можно так:
    experiment.readings_from_reports(sol.reports(noise=True))
или функцией core_readings() ниже. Если readings не передан,
на «дисплеях» стоит прочерк.

Запуск:
    python circuit_view.py           — сохранить картинки в results/
    python circuit_view.py --demo    — окно со слайдерами R, U и выбором схемы
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

# ---------------------------------------------------------------- стиль
WIRE = "#1f2328"          # провода и элементы
CURRENT = "#c0392b"       # стрелки токов
VBRANCH = "#1f6fb2"       # ветвь вольтметра
READOUT_BG = "#fdf6e3"    # фон табличек с показаниями
LW = 2.0                  # толщина линий


# ======================================================================
#  Показания приборов — из ядра (сама схема ничего не считает)
# ======================================================================
def core_readings(scheme="A", U_src=10.0, R=100.0, R_A=0.5, R_V=1e6,
                  noise=False, seed=None, **params):
    """Собрать схему А или Б на элементах ядра (presets.py), решить её
    ядром и вернуть показания приборов {'U_meas': ..., 'I_meas': ...}.
    None вместо числа — прибор перегружен."""
    import presets
    from experiment import measure_once

    circuit = presets.build(scheme, R=R, emf=U_src, R_A=R_A, R_V=R_V, **params)
    readings, _warnings = measure_once(circuit, U_src, seed=seed, noise=noise)
    return readings


def _with_r_meas(readings):
    """Копия показаний + R_meas = U_meas / I_meas (если оба есть)."""
    rd = dict(readings or {})
    U, I = rd.get("U_meas"), rd.get("I_meas")
    rd["R_meas"] = U / I if U is not None and I else None
    return rd


def _reading_text(value, unit):
    return "перегрузка" if value is None else fmt(value, unit)


# ======================================================================
#  Форматирование чисел с приставками (Ом/кОм/МОм, А/мА/мкА)
# ======================================================================
_PREFIXES = [(1e6, "М"), (1e3, "к"), (1, ""), (1e-3, "м"), (1e-6, "мк")]


def fmt(value, unit, digits=4):
    """fmt(0.00123, 'А') -> '1,230 мА'  (запятая — как принято в РФ).
    Для «Ом» дольные приставки не используются: 0,5 Ом, а не 500 мОм."""
    if value == 0:
        return f"0 {unit}"
    if value in (float("inf"), float("-inf")):      # идеальный вольтметр R_V = ∞
        return f"∞ {unit}"
    for factor, prefix in _PREFIXES:
        if abs(value) >= factor or (unit == "Ом" and factor == 1):
            break
    v = value / factor
    decimals = max(0, digits - 1 - len(str(int(abs(v)))) + 1)
    s = f"{v:.{decimals}f}".replace(".", ",")
    return f"{s} {prefix}{unit}"


# ======================================================================
#  Элементы схемы
# ======================================================================
def wire(ax, *pts, color=WIRE, lw=LW):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=1)


def node(ax, x, y, color=WIRE):
    """Точка соединения проводов (узел)."""
    ax.add_patch(Circle((x, y), 0.09, color=color, zorder=5))


def resistor_h(ax, x1, x2, y, label=None, color=WIRE, label_above=True,
               w=1.2, h=0.42):
    """Горизонтальный резистор по ГОСТ: прямоугольник 10×4 мм (масштаб)."""
    xc = (x1 + x2) / 2
    wire(ax, (x1, y), (xc - w / 2, y), color=color)
    wire(ax, (xc + w / 2, y), (x2, y), color=color)
    ax.add_patch(Rectangle((xc - w / 2, y - h / 2), w, h, fill=True,
                           facecolor="white", edgecolor=color, lw=LW,
                           zorder=3))
    if label:
        dy = h / 2 + 0.28 if label_above else -(h / 2 + 0.32)
        ax.text(xc, y + dy, label, ha="center",
                va="bottom" if label_above else "top", fontsize=11,
                color=color, zorder=6)


def meter(ax, x, y, letter, color=WIRE, r=0.45):
    """Измерительный прибор: кружок с буквой (A — амперметр, V — вольтметр)."""
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                        lw=LW, zorder=3))
    ax.text(x, y, letter, ha="center", va="center", fontsize=15,
            fontweight="bold", color=color, zorder=4)
    return r


def source_v(ax, x, y_bot, y_top, label=None):
    """Источник постоянного напряжения на вертикальном проводе.
    Длинная тонкая пластина — «+», короткая толстая — «−»."""
    yc = (y_bot + y_top) / 2
    gap = 0.14
    wire(ax, (x, y_top), (x, yc + gap))
    wire(ax, (x, y_bot), (x, yc - gap))
    ax.plot([x - 0.45, x + 0.45], [yc + gap] * 2, color=WIRE, lw=LW,
            zorder=3)                                         # «+»
    ax.plot([x - 0.22, x + 0.22], [yc - gap] * 2, color=WIRE, lw=LW * 2.6,
            solid_capstyle="butt", zorder=3)                  # «−»
    ax.text(x + 0.55, yc + gap + 0.08, "+", fontsize=12, color=WIRE)
    ax.text(x + 0.55, yc - gap - 0.42, "−", fontsize=12, color=WIRE)
    if label:
        ax.text(x - 0.6, yc, label, ha="right", va="center", fontsize=12.5)


def switch_h(ax, x1, x2, y, label="K"):
    """Ключ (выключатель) в замкнутом положении на горизонтальном проводе."""
    a, b = x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7
    wire(ax, (x1, y), (a, y))
    wire(ax, (b, y), (x2, y))
    ax.plot([a, b - 0.05], [y, y + 0.32], color=WIRE, lw=LW, zorder=3)
    node(ax, a, y)
    ax.add_patch(Circle((b, y), 0.08, facecolor="white", edgecolor=WIRE,
                        lw=1.5, zorder=5))
    ax.text((a + b) / 2, y + 0.45, label, ha="center", va="bottom",
            fontsize=11)


def current_arrow(ax, x1, y1, x2, y2, label, label_pos="above"):
    """Стрелка направления тока поверх провода с подписью."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=CURRENT, lw=2.0,
                                mutation_scale=16))
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    off = {"above": (0, 0.22, "center", "bottom"),
           "below": (0, -0.22, "center", "top"),
           "left": (-0.2, 0, "right", "center"),
           "right": (0.2, 0, "left", "center")}[label_pos]
    ax.text(xm + off[0], ym + off[1], label, color=CURRENT, fontsize=11,
            ha=off[2], va=off[3], zorder=6)


def readout(ax, x, y, text, color=WIRE):
    """Табличка с показанием прибора («дисплей»)."""
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5,
            family="DejaVu Sans Mono", color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", fc=READOUT_BG, ec=color,
                      lw=1.2))


def dashed_box(ax, x, y, w, h, label, color, below=False):
    """Пунктирная рамка вокруг «реального прибора» в эквивалентной схеме."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fill=False, ls="--", lw=1.2, ec=color,
                                zorder=2))
    if below:
        ax.text(x + w / 2, y - 0.12, label, ha="center", va="top",
                fontsize=9.5, color=color)
    else:
        ax.text(x + w / 2, y + h + 0.12, label, ha="center", va="bottom",
                fontsize=9.5, color=color)


# ======================================================================
#  Главная функция: схема целиком
# ======================================================================
def draw_circuit(ax, scheme="A", U_src=10.0, R=100.0, R_A=0.5, R_V=1e6,
                 readings=None, equivalent=False, show_formula=False,
                 title=True, show_readings=False, load_label=None):
    """Рисует схему А или Б на осях ax.

    show_readings — показывать «дисплеи» приборов (I, U) и подписи R_A, R_V.
                    По умолчанию выключено: на схеме ключ разомкнут.

    readings — словарь {'U_meas':..., 'I_meas':...} с показаниями приборов
               ядра (с шумом или без). Схема сама их не считает: если
               readings не передан, на «дисплеях» прочерк.
    equivalent — заменить приборы их внутренними сопротивлениями.
    load_label — подпись нагрузки вместо «R = …» (для дополнительных схем:
                 два резистора, лампа).

    Возвращает показания (readings + R_meas) или пустой словарь.
    """
    ax.clear()
    rd = _with_r_meas(readings) if readings else {}

    Y_TOP, Y_BOT, Y_V = 4.0, 0.0, 6.2
    X_L, X_R = 0.0, 10.0

    # ---- нижний и правый провода (ключ теперь сверху, до амперметра)
    wire(ax, (X_L, Y_BOT), (X_R, Y_BOT), (X_R, Y_TOP), (8.0, Y_TOP))
    node(ax, 8.0, Y_TOP)

    # ---- источник ЭДС слева
    source_v(ax, X_L, Y_BOT, Y_TOP,
             label=r"$\mathcal{E}$ = " + fmt(U_src, "В", 3))

    # ---- ключ K сразу после источника, до амперметра
    X_K1, X_K2 = 0.15, 1.35
    wire(ax, (X_L, Y_TOP), (X_K1, Y_TOP))
    switch_h(ax, X_K1, X_K2, Y_TOP)

    # ---- положения элементов для схем А и Б
    if scheme == "A":
        x_amm, x_vnode = 2.4, 4.0          # ключ → амперметр → узел
        wire(ax, (X_K2, Y_TOP), (x_amm - 0.45, Y_TOP))
        wire(ax, (x_amm + 0.45, Y_TOP), (x_vnode, Y_TOP))
    else:
        x_vnode, x_amm = 1.9, 3.3          # ключ → узел → амперметр
        wire(ax, (X_K2, Y_TOP), (x_vnode, Y_TOP), (x_amm - 0.45, Y_TOP))
        wire(ax, (x_amm + 0.45, Y_TOP), (4.6, Y_TOP))
    node(ax, x_vnode, Y_TOP)

    # ---- резистор R
    x_r1 = x_vnode if scheme == "A" else 4.6
    resistor_h(ax, x_r1, 8.0, Y_TOP, label_above=False,
               label=load_label or f"R = {fmt(R, 'Ом', 3)}")

    # ---- амперметр (или R_A)
    if equivalent:
        resistor_h(ax, x_amm - 0.45, x_amm + 0.45, Y_TOP, w=0.7, h=0.32,
                   color=WIRE)
        dashed_box(ax, x_amm - 0.65, Y_TOP - 0.4, 1.3, 0.8,
                   r"амперметр: $R_A$", WIRE, below=True)
    else:
        meter(ax, x_amm, Y_TOP, "A")
        if show_readings:
            ax.text(x_amm, Y_TOP - 0.62, f"$R_A$={fmt(R_A, 'Ом', 2)}",
                    ha="center", va="top", fontsize=8.5, color="#555")
    if show_readings:
        readout(ax, x_amm, Y_TOP + 0.95,
                "I = " + (_reading_text(rd["I_meas"], "А") if rd else "—"))

    # ---- ветвь вольтметра (или R_V)
    x_vm = (x_vnode + 8.0) / 2
    wire(ax, (x_vnode, Y_TOP), (x_vnode, Y_V), (x_vm - 0.45, Y_V),
         color=VBRANCH)
    wire(ax, (x_vm + 0.45, Y_V), (8.0, Y_V), (8.0, Y_TOP), color=VBRANCH)
    if equivalent:
        resistor_h(ax, x_vm - 0.45, x_vm + 0.45, Y_V, w=0.7, h=0.32,
                   color=VBRANCH)
        dashed_box(ax, x_vm - 0.65, Y_V - 0.4, 1.3, 0.8,
                   r"вольтметр: $R_V$", VBRANCH, below=True)
    else:
        meter(ax, x_vm, Y_V, "V", color=VBRANCH)
        if show_readings:
            ax.text(x_vm, Y_V - 0.62, f"$R_V$={fmt(R_V, 'Ом', 2)}",
                    ha="center", va="top", fontsize=8.5, color=VBRANCH)
    if show_readings:
        readout(ax, x_vm, Y_V + 0.95,
                "U = " + (_reading_text(rd["U_meas"], "В") if rd else "—"),
                color=VBRANCH)

    # ---- стрелки токов
    current_arrow(ax, X_L, 2.75, X_L, 3.75, r"$I$", "right")
    if scheme == "A":
        current_arrow(ax, 4.15, Y_TOP, 5.0, Y_TOP, r"$I_R$", "below")
    else:
        current_arrow(ax, 4.0, Y_TOP, 4.6, Y_TOP, r"$I_R$", "below")
    current_arrow(ax, x_vnode, 4.55, x_vnode, 5.6, r"$I_V$",
                  "right" if scheme == "A" else "left")
    current_arrow(ax, X_R, 2.6, X_R, 1.4, r"$I$", "right")

    # ---- фигурная «скобка», показывающая, что меряет вольтметр
    xb1 = x_vnode + 0.05
    ax.annotate("", xy=(xb1, 3.0), xytext=(7.95, 3.0),
                arrowprops=dict(arrowstyle="<->", color=VBRANCH, lw=1.2,
                                ls="--"))
    span = r"$U_V = U_R$" if scheme == "A" else r"$U_V = U_A + U_R$"
    ax.text((xb1 + 7.95) / 2, 2.75, span, ha="center", va="top",
            fontsize=10.5, color=VBRANCH)

    # ---- заголовок и формулы
    if title:
        name = {"A": "Схема А — вольтметр параллельно резистору",
                "B": "Схема Б — вольтметр параллельно «амперметр + R»"}
        ax.set_title(name[scheme] + (" (эквивалентная)" if equivalent
                                     else ""), fontsize=12.5, pad=10,
                     fontweight="bold")

    if show_formula:
        if scheme == "A":
            f1 = r"$I_{\rm изм} = I_R + I_V$"
            f2 = (r"$R_{\rm изм} = \dfrac{R\,R_V}{R+R_V} < R$")
        else:
            f1 = r"$U_{\rm изм} = U_R + U_A$"
            f2 = r"$R_{\rm изм} = R + R_A > R$"
        txt = f1 + "        " + f2
        if rd.get("R_meas") is not None:      # число — только из показаний ядра
            delta = (rd["R_meas"] - R) / R * 100
            txt += ("\n" + r"$R_{\rm изм} = U/I$ = " + fmt(rd["R_meas"], "Ом", 5)
                    + f"    отклонение δ = {delta:+.3f} %".replace(".", ","))
        ax.text(5.0, -0.95, txt, ha="center", va="top", fontsize=11,
                linespacing=1.9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8",
                          ec="#c9d1d9"))

    ax.set_xlim(-2.4, 11.2)
    ax.set_ylim(-3.1 if show_formula else -0.9, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return rd


# ======================================================================
#  Готовые рисунки
# ======================================================================
def save_figures(out_dir="results", U_src=10.0, R=1e5, R_A=0.5, R_V=1e6):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for equiv, fname in [(False, "circuit_schemes.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
        for ax, sch in zip(axes, ["A", "B"]):
            draw_circuit(ax, sch, U_src, R, R_A, R_V, equivalent=equiv)
        fig.suptitle(
            r"Закон Ома для участка цепи:   $\mathcal{E}$ = " + fmt(U_src, "В", 3)
            + ",   R = " + fmt(R, "Ом", 3) + r",   $R_A$ = " + fmt(R_A, "Ом", 2)
            + r",   $R_V$ = " + fmt(R_V, "Ом", 2), fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    return paths


def demo():
    """Интерактивное окно: слайдеры U и lg R, переключатели схемы и вида."""
    from matplotlib.widgets import Slider, RadioButtons

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_axes([0.03, 0.22, 0.94, 0.75])
    ax_u = fig.add_axes([0.25, 0.12, 0.55, 0.03])
    ax_r = fig.add_axes([0.25, 0.07, 0.55, 0.03])
    ax_s = fig.add_axes([0.03, 0.02, 0.15, 0.12])
    ax_e = fig.add_axes([0.83, 0.02, 0.15, 0.12])

    s_u = Slider(ax_u, "ЭДС, В", 0.5, 10.0, valinit=5.0, valstep=0.5)
    s_r = Slider(ax_r, "lg R (R в Ом)", 0, 6, valinit=2, valstep=0.1)
    rb_s = RadioButtons(ax_s, ["Схема А", "Схема Б"])
    rb_e = RadioButtons(ax_e, ["Приборы", "Эквивалент"])

    def update(_=None):
        draw_circuit(ax, "A" if rb_s.value_selected == "Схема А" else "B",
                     U_src=s_u.val, R=10 ** s_r.val,
                     equivalent=rb_e.value_selected == "Эквивалент")
        fig.canvas.draw_idle()

    for w in (s_u, s_r):
        w.on_changed(update)
    for w in (rb_s, rb_e):
        w.on_clicked(update)
    update()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--demo", action="store_true",
                   help="интерактивное окно со слайдерами")
    p.add_argument("--R", type=float, default=1e5, help="сопротивление, Ом")
    p.add_argument("--U", type=float, default=10.0, help="напряжение, В")
    args = p.parse_args()
    if args.demo:
        demo()
    else:
        for path in save_figures(R=args.R, U_src=args.U):
            print("сохранено:", path)
