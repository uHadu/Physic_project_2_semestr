"""
components.py — шаблоны элементов цепи.

Каждый элемент знает:
  * свои выводы (terminals) и параметры (params) — по ним интерфейс может
    сам построить форму ввода;
  * как «вписать» себя в систему уравнений узлового метода (stamp);
  * что про себя сообщить после расчёта (report): U, I, P и специфичное.
"""
from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass
from typing import Any, ClassVar

GROUND_NAMES = frozenset({"0", "gnd", "GND", "земля"})
ALPHA_TUNGSTEN = 4.5e-3  # ТКС вольфрама, 1/К


class ElementError(ValueError):
    """Неверные параметры элемента."""


# ---------------------------------------------------------------------------
# Описание параметров (для интерфейса и проверки)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    key: str
    title: str
    unit: str = ""
    default: Any = None
    minimum: float | None = None       # нижняя граница (включительно)
    positive: bool = False             # строго > 0
    kind: str = "float"                # float | bool | list | str


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on", "да", "вкл", "замкнут", "closed"}:
        return True
    if s in {"0", "false", "no", "off", "нет", "выкл", "разомкнут", "open"}:
        return False
    raise ElementError(f"Не понимаю логическое значение «{v}»")


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------

class Element:
    kind: ClassVar[str] = ""
    title: ClassVar[str] = ""
    terminals: ClassVar[tuple[str, ...]] = ("a", "b")
    params: ClassVar[tuple[ParamSpec, ...]] = ()
    presets: ClassVar[dict[str, dict]] = {}
    n_branches: ClassVar[int] = 0      # сколько доп. неизвестных (токов ветвей) нужно
    nonlinear: ClassVar[bool] = False

    def __init__(self, name: str, nodes: list[str] | tuple[str, ...], model: str | None = None, **kw):
        if not name:
            raise ElementError("У элемента должно быть имя")
        self.name = str(name)
        nodes = [str(n) for n in nodes]
        if len(nodes) != len(self.terminals):
            raise ElementError(
                f"{self.name}: нужно {len(self.terminals)} узла(ов) "
                f"({', '.join(self.terminals)}), передано {len(nodes)}"
            )
        self.nodes = nodes
        self.model = model
        self.branch_index: int | None = None  # назначает решатель

        values = {p.key: p.default for p in self.params}
        if model is not None:
            if model not in self.presets:
                raise ElementError(
                    f"{self.name}: неизвестная модель «{model}». Есть: {', '.join(self.presets) or '—'}"
                )
            values.update(self.presets[model])
        known = {p.key for p in self.params}
        for k, v in kw.items():
            if k not in known:
                raise ElementError(f"{self.name}: неизвестный параметр «{k}». Есть: {', '.join(known)}")
            values[k] = v
        for p in self.params:
            setattr(self, p.key, self._check(p, values[p.key]))

    def _check(self, p: ParamSpec, v: Any) -> Any:
        if v is None:
            return None
        if p.kind == "bool":
            return _to_bool(v)
        if p.kind == "list":
            vals = tuple(sorted(float(x) for x in v))
            if not vals or vals[0] <= 0:
                raise ElementError(f"{self.name}.{p.key}: нужен список положительных чисел")
            return vals
        if p.kind == "str":
            return str(v)
        v = float(v)
        if math.isnan(v):
            raise ElementError(f"{self.name}.{p.key}: NaN недопустим")
        if p.positive and not v > 0:
            raise ElementError(f"{self.name}.{p.key} = {v}: должно быть > 0")
        if p.minimum is not None and v < p.minimum:
            raise ElementError(f"{self.name}.{p.key} = {v}: должно быть ≥ {p.minimum}")
        return v

    def set_param(self, key: str, value: Any) -> None:
        for p in self.params:
            if p.key == key:
                setattr(self, key, self._check(p, value))
                return
        raise ElementError(f"{self.name}: нет параметра «{key}»")

    # --- для решателя ------------------------------------------------------
    def reset(self, sys) -> None:
        """Сброс внутреннего состояния перед новым расчётом (нужно нелинейным)."""

    def stamp(self, sys) -> None:
        raise NotImplementedError

    def current(self, sol) -> float:
        """Ток через элемент от первого вывода ко второму."""
        raise NotImplementedError

    def report(self, sol, rng: random.Random | None = None, noise: bool = False) -> dict:
        U = sol.V(self.nodes[0]) - sol.V(self.nodes[1])
        I = self.current(sol)
        return {"U": U, "I": I, "P": U * I}

    # --- сериализация ------------------------------------------------------
    def to_spec(self) -> dict:
        d = {"type": self.kind, "name": self.name, "nodes": list(self.nodes)}
        for p in self.params:
            v = getattr(self, p.key)
            d[p.key] = list(v) if isinstance(v, tuple) else v
        return d

    @classmethod
    def describe(cls) -> dict:
        """Описание шаблона для интерфейса: выводы, параметры, модели."""
        return {
            "type": cls.kind,
            "title": cls.title,
            "terminals": list(cls.terminals),
            "params": [p.__dict__ for p in cls.params],
            "presets": list(cls.presets),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.nodes})"


# ---------------------------------------------------------------------------
# Пассивные элементы
# ---------------------------------------------------------------------------

class Resistor(Element):
    kind = "resistor"
    title = "Резистор"
    params = (
        ParamSpec("R", "Сопротивление", "Ом", 1000.0, positive=True),
        ParamSpec("P_max", "Допустимая мощность", "Вт", 0.25, positive=True),
    )

    def stamp(self, sys):
        sys.conductance(self.nodes[0], self.nodes[1], 1.0 / self.R)

    def current(self, sol):
        return (sol.V(self.nodes[0]) - sol.V(self.nodes[1])) / self.R

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        r["overheated"] = self.P_max is not None and abs(r["P"]) > self.P_max
        return r


class Switch(Element):
    kind = "switch"
    title = "Ключ"
    params = (
        ParamSpec("closed", "Замкнут", "", True, kind="bool"),
        ParamSpec("R_on", "Сопротивление замкнутого ключа", "Ом", 1e-3, positive=True),
    )

    def stamp(self, sys):
        if self.closed:
            sys.conductance(self.nodes[0], self.nodes[1], 1.0 / self.R_on)

    def current(self, sol):
        if not self.closed:
            return 0.0
        return (sol.V(self.nodes[0]) - sol.V(self.nodes[1])) / self.R_on

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        r["closed"] = self.closed
        return r


class Lamp(Element):
    """
    Лампа накаливания (вольфрамовая нить).

    Сопротивление растёт с температурой: R = R0·(1 + α·ΔT),
    нагрев определяется тепловым балансом ΔT = P / G_th, P = U²/R.
    Отсюда R² − R0·R − R0·a·U² = 0, a = α/G_th, и
        R(U) = (R0 + √(R0² + 4·R0·a·U²)) / 2.
    Параметры задаются через номинал: U_nom, P_nom и отношение
    сопротивлений горячей и холодной нити cold_ratio = R_гор/R0 (у вольфрама ≈ 10–15).
    Тогда a = (cold_ratio − 1) / P_nom.
    """
    kind = "lamp"
    title = "Лампа накаливания"
    nonlinear = True
    params = (
        ParamSpec("U_nom", "Номинальное напряжение", "В", 6.0, positive=True),
        ParamSpec("P_nom", "Номинальная мощность", "Вт", 3.0, positive=True),
        ParamSpec("cold_ratio", "R_гор / R_хол", "", 10.0, minimum=1.0),
        ParamSpec("burn_factor", "Перегорает при P > k·P_nom", "", 1.5, minimum=1.0),
        ParamSpec("T0", "Температура среды", "К", 293.0, positive=True),
        ParamSpec("burned", "Перегорела", "", False, kind="bool"),
    )
    presets = {
        "6V_3W": {"U_nom": 6.0, "P_nom": 3.0},
        "12V_5W": {"U_nom": 12.0, "P_nom": 5.0},
        "3.5V_0.26A": {"U_nom": 3.5, "P_nom": 3.5 * 0.26},   # МН 3,5-0,26 (фонарик)
        "2.5V_0.3A": {"U_nom": 2.5, "P_nom": 2.5 * 0.3},     # МН 2,5-0,3 (школьная)
        "230V_60W": {"U_nom": 230.0, "P_nom": 60.0},
    }

    @property
    def R_hot(self) -> float:
        return self.U_nom ** 2 / self.P_nom

    @property
    def R0(self) -> float:
        return self.R_hot / self.cold_ratio

    @property
    def _a(self) -> float:
        return (self.cold_ratio - 1.0) / self.P_nom

    def resistance(self, U: float) -> float:
        R0 = self.R0
        return 0.5 * (R0 + math.sqrt(R0 * R0 + 4.0 * R0 * self._a * U * U))

    def i_and_g(self, U: float) -> tuple[float, float]:
        """Ток и дифференциальная проводимость dI/dU."""
        R0, a = self.R0, self._a
        s = math.sqrt(R0 * R0 + 4.0 * R0 * a * U * U)
        R = 0.5 * (R0 + s)
        dR = 2.0 * R0 * a * U / s
        return U / R, (R - U * dR) / (R * R)

    def stamp(self, sys):
        if self.burned:
            return
        a, b = self.nodes
        U0 = sys.v(a) - sys.v(b)
        I0, g = self.i_and_g(U0)
        sys.conductance(a, b, g)
        sys.current(a, b, I0 - g * U0)

    def current(self, sol):
        if self.burned:
            return 0.0
        return self.i_and_g(sol.V(self.nodes[0]) - sol.V(self.nodes[1]))[0]

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        U = r["U"]
        R = math.inf if self.burned else self.resistance(U)
        dT = (R / self.R0 - 1.0) / ALPHA_TUNGSTEN if not self.burned else 0.0
        r.update(
            R=R,
            T=self.T0 + dT,                          # температура нити, К
            brightness=0.0 if self.burned else min(abs(r["P"]) / self.P_nom, 2.0),  # доля номинала
            burned=self.burned,
            will_burn=not self.burned and abs(r["P"]) > self.burn_factor * self.P_nom,
        )
        return r


# ---------------------------------------------------------------------------
# Источник
# ---------------------------------------------------------------------------

class VoltageSource(Element):
    """Источник ЭДС с внутренним сопротивлением. Выводы: «+», «−»."""
    kind = "source"
    title = "Источник ЭДС"
    terminals = ("+", "-")
    n_branches = 1
    params = (
        ParamSpec("emf", "ЭДС", "В", 10.0),
        ParamSpec("r", "Внутреннее сопротивление", "Ом", 0.0, minimum=0.0),
    )
    presets = {
        "батарейка_AA": {"emf": 1.5, "r": 0.15},
        "крона_9V": {"emf": 9.0, "r": 1.5},
        "лабораторный": {"emf": 10.0, "r": 0.0},
    }

    def stamp(self, sys):
        # ток ветви i течёт внутрь «+» и выходит из «−»; v+ − v− − r·i = ЭДС
        sys.branch(self.nodes[0], self.nodes[1], self.branch_index, self.r, self.emf)

    def current(self, sol):
        return -float(sol.x[self.branch_index])  # отдаваемый во внешнюю цепь ток

    def report(self, sol, rng=None, noise=False):
        U = sol.V(self.nodes[0]) - sol.V(self.nodes[1])   # напряжение на клеммах
        I = self.current(sol)
        return {"U": U, "I": I, "P": U * I, "P_internal": I * I * self.r}


# ---------------------------------------------------------------------------
# Пользовательский элемент: свои параметры + ВАХ формулой или таблицей
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10,
    "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
    "sinh": math.sinh, "cosh": math.cosh, "atan": math.atan,
    "sign": lambda x: math.copysign(1.0, x) if x else 0.0,
    "min": min, "max": max,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_RESERVED = {"U", *_ALLOWED_FUNCS, *_ALLOWED_CONSTS}


class _FloatConstants(ast.NodeTransformer):
    """Все числа в формуле — float: 10**10**10 даст OverflowError, а не зависание."""
    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ElementError(f"В формуле допустимы только числа, а не «{node.value!r}»")
        return ast.copy_location(ast.Constant(float(node.value)), node)


def compile_formula(text: str, param_names: set[str]):
    """
    Безопасный разбор формулы I(U). Разрешены: + − * / **, скобки, числа,
    переменная U, имена параметров элемента, функции из _ALLOWED_FUNCS, pi, e.
    Никаких атрибутов, индексов, лямбд и прочего — проверяется по дереву выражения.
    """
    src = text.replace("^", "**").strip()
    if len(src) > 500:
        raise ElementError("Формула слишком длинная (больше 500 символов)")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ElementError(f"Ошибка в формуле «{text}»: {exc.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ElementError(f"В формуле нельзя использовать «{type(node).__name__}»")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS or node.keywords:
                raise ElementError(
                    f"Неизвестная функция в формуле. Можно: {', '.join(sorted(_ALLOWED_FUNCS))}")
        if isinstance(node, ast.Name) and not (
                node.id in _RESERVED or node.id in param_names):
            raise ElementError(f"В формуле неизвестное имя «{node.id}» — это не U и не параметр элемента")
    tree = ast.fix_missing_locations(_FloatConstants().visit(tree))
    code = compile(tree, "<formula>", "eval")
    env_base = {"__builtins__": {}, **_ALLOWED_FUNCS, **_ALLOWED_CONSTS}

    def f(U: float, params: dict[str, float]) -> float:
        return float(eval(code, env_base, {**params, "U": float(U)}))  # noqa: S307 — дерево проверено выше

    return f


def _parse_table(points) -> list[tuple[float, float]]:
    """[[U, I], ...] или строка '0:0, 1:0.01, 2:0.03' → отсортированный список точек."""
    if isinstance(points, str):
        pts = []
        for chunk in points.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ElementError(f"Точку таблицы пишите как U:I, а не «{chunk}»")
            u, i = chunk.split(":", 1)
            pts.append((_num(u), _num(i)))
    else:
        pts = [(float(u), float(i)) for u, i in points]
    pts.sort()
    if len(pts) < 2:
        raise ElementError("В таблице ВАХ нужно минимум 2 точки")
    if any(pts[k][0] == pts[k + 1][0] for k in range(len(pts) - 1)):
        raise ElementError("В таблице ВАХ повторяются значения U")
    return pts


def _num(s: str) -> float:
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        raise ElementError(f"Не число: «{s}»") from None


class CustomElement(Element):
    """
    Пользовательский двухполюсник. Пользователь сам решает, какими параметрами
    обладает элемент, и задаёт его ВАХ одним из способов:

      * formula — ток как функция напряжения, с его параметрами:
            custom X1 a b formula="U / (R0 * (1 + k*U**2))" R0=10 k=0.01
      * table   — точки (U, I), между ними — линейная интерполяция,
                  за краями — продолжение крайнего отрезка:
            custom X1 a b table="0:0, 1:0.02, 2:0.03, 4:0.04"
        Если в таблице только U ≥ 0, ВАХ считается симметричной: I(−U) = −I(U).

    Пассивный элемент должен давать I(0) = 0 — иначе это скрытый источник,
    об этом выдаётся предупреждение.
    """
    kind = "custom"
    title = "Пользовательский элемент"
    nonlinear = True
    params = (
        ParamSpec("formula", "ВАХ: формула I(U)", "А", None, kind="str"),
        ParamSpec("table", "ВАХ: таблица точек U:I", "", None, kind="table"),
        ParamSpec("label", "Название (для отображения)", "", None, kind="str"),
    )

    def __init__(self, name, nodes, model=None, **kw):
        own = {p.key for p in self.params}
        user = {k: v for k, v in kw.items() if k not in own}
        for k in user:
            if not k.isidentifier() or k in _RESERVED:
                raise ElementError(f"{name}: «{k}» нельзя использовать как имя параметра")
        self.user_params: dict[str, float] = {}
        super().__init__(name, nodes, model, **{k: v for k, v in kw.items() if k in own})
        for k, v in user.items():
            try:
                self.user_params[k] = float(v)
            except (TypeError, ValueError):
                raise ElementError(f"{name}.{k} = «{v}»: параметр должен быть числом") from None
        self._rebuild()

    def _check(self, p, v):
        if p.kind == "table":
            return None if v is None else _parse_table(v)
        return super()._check(p, v)

    def _rebuild(self):
        if (self.formula is None) == (self.table is None):
            raise ElementError(f"{self.name}: задайте ВАХ ровно одним способом — formula или table")
        self.warnings: list[str] = []
        if self.formula is not None:
            self._f = compile_formula(self.formula, set(self.user_params))
            bad, vals = [], {}
            for u in (-10.0, -1.0, 0.0, 1.0, 10.0):
                try:
                    v = self._f(u, self.user_params)
                except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                    v = math.nan
                if math.isfinite(v):
                    vals[u] = v
                else:
                    bad.append(u)
            if 0.0 in bad or 1.0 in bad:
                raise ElementError(f"{self.name}: формула «{self.formula}» не считается при U = 0 или 1 В "
                                   "(деление на ноль, корень из отрицательного, переполнение?)")
            if bad:
                self.warnings.append(f"{self.name}: формула не определена при U = "
                                     f"{', '.join(f'{u:g}' for u in bad)} В — подайте напряжение другой полярности "
                                     "или используйте abs(U) и sign(U)")
            if abs(vals[0.0]) > 1e-12:
                self.warnings.append(f"{self.name}: I(0) = {vals[0.0]:g} А ≠ 0 — элемент ведёт себя как источник")
        else:
            self._symmetric = self.table[0][0] >= 0
            if self.table[0][0] == 0 and abs(self.table[0][1]) > 1e-12:
                self.warnings.append(f"{self.name}: I(0) ≠ 0 — элемент ведёт себя как источник")

    def set_param(self, key, value):
        if key in self.user_params:
            self.user_params[key] = float(value)
            self._rebuild()
            return
        super().set_param(key, value)
        self._rebuild()

    # --- ВАХ ---------------------------------------------------------------
    def _table_i_g(self, U: float) -> tuple[float, float]:
        sign = 1.0
        if self._symmetric and U < 0:
            U, sign = -U, -1.0
        pts = self.table
        k = 0
        while k < len(pts) - 2 and U > pts[k + 1][0]:
            k += 1
        (u0, i0), (u1, i1) = pts[k], pts[k + 1]
        g = (i1 - i0) / (u1 - u0)
        return sign * (i0 + g * (U - u0)), g

    def i_and_g(self, U: float) -> tuple[float, float]:
        if self.table is not None:
            return self._table_i_g(U)
        p = self.user_params
        h = 1e-6 * max(1.0, abs(U))
        errs = (ZeroDivisionError, ValueError, OverflowError, TypeError)
        try:
            I = self._f(U, p)
        except errs:
            raise ElementError(f"формула не считается при U = {U:g} В") from None
        g = None
        for a, b in ((U + h, U - h), (U + h, U), (U, U - h)):   # центральная, потом односторонние
            try:
                g = (self._f(a, p) - self._f(b, p)) / (a - b)
                break
            except errs:
                continue
        if g is None:
            raise ElementError(f"не удаётся найти dI/dU при U = {U:g} В")
        if not (math.isfinite(I) and math.isfinite(g)):
            raise ElementError(f"формула даёт бесконечность при U = {U:g} В")
        return I, g

    def stamp(self, sys):
        a, b = self.nodes
        U0 = sys.v(a) - sys.v(b)
        I0, g = self.i_and_g(U0)
        g_lin = max(g, 1e-12)          # для устойчивости Ньютона на участках с dI/dU ≤ 0
        sys.conductance(a, b, g_lin)
        sys.current(a, b, I0 - g_lin * U0)

    def current(self, sol):
        return self.i_and_g(sol.V(self.nodes[0]) - sol.V(self.nodes[1]))[0]

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        U, I = r["U"], r["I"]
        _, g = self.i_and_g(U)
        r["R_static"] = U / I if I else math.inf     # U/I — «сопротивление» в точке
        r["R_diff"] = 1.0 / g if g else math.inf     # dU/dI — дифференциальное
        r.update({f"param.{k}": v for k, v in self.user_params.items()})
        return r

    def to_spec(self):
        d = super().to_spec()
        if d.get("table") is not None:
            d["table"] = [list(pt) for pt in d["table"]]
        d.update(self.user_params)
        return d

    @classmethod
    def describe(cls):
        d = super().describe()
        d["free_params"] = True   # кроме formula/table можно добавлять любые свои числовые параметры
        d["formula_functions"] = sorted(_ALLOWED_FUNCS) + sorted(_ALLOWED_CONSTS)
        return d


# ---------------------------------------------------------------------------
# Измерительные приборы
# ---------------------------------------------------------------------------

def _meter_reading(el, true_value: float, rng: random.Random | None, noise: bool) -> dict:
    """
    Цифровой прибор: автовыбор предела, Δ = γ·предел/100 %,
    случайная ошибка равномерна в [−Δ; +Δ], округление до разрешения предел/counts.
    """
    limit = next((r for r in el.ranges if abs(true_value) <= r), None)
    if limit is None:
        return {"reading": None, "error": None, "limit": el.ranges[-1], "overload": True}
    delta = el.accuracy_class * limit / 100.0
    resolution = limit / el.counts
    value = true_value
    if noise:
        value += (rng or random).uniform(-delta, delta)
    value = round(value / resolution) * resolution
    value = round(value, max(0, -math.floor(math.log10(resolution)) + 1))
    return {"reading": value, "error": delta, "limit": limit,
            "resolution": resolution, "overload": False}


_METER_COMMON = (
    ParamSpec("accuracy_class", "Класс точности", "%", 0.5, positive=True),
    ParamSpec("counts", "Разрядность (отсчётов на предел)", "", 2000, positive=True),
)


class Ammeter(Element):
    """Амперметр: R_A последовательно, показывает ток от «+» к «−». R_A = 0 — идеальный."""
    kind = "ammeter"
    title = "Амперметр"
    terminals = ("+", "-")
    n_branches = 1
    params = (
        ParamSpec("R_A", "Внутреннее сопротивление", "Ом", 0.5, minimum=0.0),
        ParamSpec("ranges", "Пределы", "А", (200e-6, 2e-3, 20e-3, 200e-3, 2.0, 10.0), kind="list"),
    ) + _METER_COMMON
    presets = {
        "идеальный": {"R_A": 0.0, "accuracy_class": 1e-9, "counts": 10 ** 9},
        "мультиметр": {"R_A": 0.5},
        "школьный": {"R_A": 0.1, "ranges": (2.0,), "accuracy_class": 2.5, "counts": 40},
    }

    def stamp(self, sys):
        sys.branch(self.nodes[0], self.nodes[1], self.branch_index, self.R_A, 0.0)

    def current(self, sol):
        return float(sol.x[self.branch_index])

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        r.update(_meter_reading(self, r["I"], rng, noise))
        return r


class Voltmeter(Element):
    """Вольтметр: R_V параллельно, показывает U(«+») − U(«−»). R_V = inf — идеальный."""
    kind = "voltmeter"
    title = "Вольтметр"
    terminals = ("+", "-")
    params = (
        ParamSpec("R_V", "Внутреннее сопротивление", "Ом", 1.0e6, positive=True),
        ParamSpec("ranges", "Пределы", "В", (0.2, 2.0, 20.0, 200.0, 1000.0), kind="list"),
    ) + _METER_COMMON
    presets = {
        "идеальный": {"R_V": math.inf, "accuracy_class": 1e-9, "counts": 10 ** 9},
        "мультиметр": {"R_V": 1.0e6},
        "цифровой_10М": {"R_V": 10.0e6},
        "школьный": {"R_V": 1.0e3, "ranges": (6.0,), "accuracy_class": 2.5, "counts": 30},
    }

    def stamp(self, sys):
        if math.isfinite(self.R_V):
            sys.conductance(self.nodes[0], self.nodes[1], 1.0 / self.R_V)

    def current(self, sol):
        if not math.isfinite(self.R_V):
            return 0.0
        return (sol.V(self.nodes[0]) - sol.V(self.nodes[1])) / self.R_V

    def report(self, sol, rng=None, noise=False):
        r = super().report(sol)
        r.update(_meter_reading(self, r["U"], rng, noise))
        return r


# ---------------------------------------------------------------------------
# Реестр шаблонов
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, type[Element]] = {
    cls.kind: cls for cls in (VoltageSource, Resistor, Lamp, CustomElement, Switch, Ammeter, Voltmeter)
}

ALIASES = {
    "источник": "source", "battery": "source", "батарея": "source", "emf": "source",
    "резистор": "resistor", "r": "resistor",
    "лампа": "lamp", "лампочка": "lamp",
    "ключ": "switch", "key": "switch", "k": "switch",
    "элемент": "custom", "свой": "custom", "пользовательский": "custom",
    "амперметр": "ammeter", "вольтметр": "voltmeter",
}


def template(kind: str) -> type[Element]:
    k = kind.strip().lower()
    k = ALIASES.get(k, k)
    if k not in TEMPLATES:
        raise ElementError(f"Неизвестный тип элемента «{kind}». Есть: {', '.join(TEMPLATES)}")
    return TEMPLATES[k]


def make(kind: str, name: str, nodes, **params) -> Element:
    """Создать элемент по названию шаблона: make("resistor", "R1", ["a", "b"], R=100)."""
    return template(kind)(name, nodes, **params)


def catalog() -> list[dict]:
    """Описание всех шаблонов — для построения палитры элементов в интерфейсе."""
    return [cls.describe() for cls in TEMPLATES.values()]