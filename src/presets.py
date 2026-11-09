"""
presets.py — готовые схемы для задания 7.1, собранные из элементов ядра.

Все схемы устроены одинаково (как на рисунке в circuit_view.py):

    «+» ЭДС ℰ (узел p) → ключ K (узел k) → амперметр A (узел a) → нагрузка → «−» (узел 0)

а вольтметр V подключается одним из двух способов:

    connection="A" — параллельно только нагрузке:          V между узлами a и 0;
    connection="B" — параллельно цепочке «амперметр + нагрузка»: V между k и 0.

Нагрузка: один резистор R (основные схемы А и Б), два резистора последовательно,
два параллельно или лампа накаливания (нелинейный контрпример).

Имена элементов во всех схемах одинаковые, по ним experiment.py снимает показания:
    E — источник, K — ключ, A — амперметр, V — вольтметр,
    R / R1, R2 — резисторы, L — лампа.

Параметры передаются в элементы ядра без изменений:
    VoltageSource(emf, r), Switch(closed, R_on),
    Ammeter(R_A, accuracy_class, counts), Voltmeter(R_V, accuracy_class, counts),
    Resistor(R, P_max), Lamp(model | U_nom, P_nom, ...).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from core_api import Ammeter, Circuit, Lamp, Resistor, Switch, VoltageSource, Voltmeter

# Имена элементов и узлов — общие для всех схем
SOURCE, SWITCH, AMMETER, VOLTMETER = "E", "K", "A", "V"
NODE_SRC, NODE_K, NODE_A, GND = "p", "k", "a", "0"

CONNECTIONS = ("A", "B")


# ---------------------------------------------------------------------------
# Общая часть: источник → ключ → амперметр, и вольтметр
# ---------------------------------------------------------------------------

def _front(emf: float, r_source: float, R_A: float, accuracy_class: float,
           counts: int, closed: bool) -> list:
    """Источник ЭДС, ключ K сразу после него и амперметр после ключа."""
    return [
        VoltageSource(SOURCE, [NODE_SRC, GND], emf=emf, r=r_source),
        Switch(SWITCH, [NODE_SRC, NODE_K], closed=closed),
        Ammeter(AMMETER, [NODE_K, NODE_A], R_A=R_A,
                accuracy_class=accuracy_class, counts=counts),
    ]


def _voltmeter(connection: str, R_V: float, accuracy_class: float, counts: int) -> Voltmeter:
    if connection not in CONNECTIONS:
        raise ValueError(f"Способ подключения вольтметра должен быть 'A' или 'B', а не {connection!r}")
    plus = NODE_A if connection == "A" else NODE_K
    return Voltmeter(VOLTMETER, [plus, GND], R_V=R_V,
                     accuracy_class=accuracy_class, counts=counts)


def _assemble(load: list, connection: str, emf: float, R_A: float, R_V: float,
              accuracy_class: float, counts: int, r_source: float, closed: bool) -> Circuit:
    return Circuit(_front(emf, r_source, R_A, accuracy_class, counts, closed)
                   + load
                   + [_voltmeter(connection, R_V, accuracy_class, counts)])


def _resistor(name: str, nodes: list[str], R: float, P_max: float | None) -> Resistor:
    kw = {} if P_max is None else {"P_max": P_max}
    return Resistor(name, nodes, R=R, **kw)


# ---------------------------------------------------------------------------
# Основные схемы задания
# ---------------------------------------------------------------------------

def scheme_a(R: float = 1e5, emf: float = 10.0, R_A: float = 0.5, R_V: float = 1e6,
             accuracy_class: float = 0.5, counts: int = 2000, r_source: float = 0.0,
             P_max: float | None = None, closed: bool = True) -> Circuit:
    """Схема А: вольтметр параллельно резистору. Ожидается R_изм = R·R_V/(R + R_V)."""
    return _assemble([_resistor("R", [NODE_A, GND], R, P_max)], "A",
                     emf, R_A, R_V, accuracy_class, counts, r_source, closed)


def scheme_b(R: float = 1e5, emf: float = 10.0, R_A: float = 0.5, R_V: float = 1e6,
             accuracy_class: float = 0.5, counts: int = 2000, r_source: float = 0.0,
             P_max: float | None = None, closed: bool = True) -> Circuit:
    """Схема Б: вольтметр параллельно «амперметр + резистор». Ожидается R_изм = R + R_A."""
    return _assemble([_resistor("R", [NODE_A, GND], R, P_max)], "B",
                     emf, R_A, R_V, accuracy_class, counts, r_source, closed)


# ---------------------------------------------------------------------------
# Дополнительные схемы
# ---------------------------------------------------------------------------

def series_resistors(R1: float = 1e3, R2: float = 2e3, connection: str = "A",
                     emf: float = 10.0, R_A: float = 0.5, R_V: float = 1e6,
                     accuracy_class: float = 0.5, counts: int = 2000,
                     r_source: float = 0.0, P_max: float | None = None,
                     closed: bool = True) -> Circuit:
    """Два резистора последовательно: R1 (a → m), R2 (m → 0). R_экв = R1 + R2."""
    load = [_resistor("R1", [NODE_A, "m"], R1, P_max), _resistor("R2", ["m", GND], R2, P_max)]
    return _assemble(load, connection, emf, R_A, R_V, accuracy_class, counts, r_source, closed)


def parallel_resistors(R1: float = 1e3, R2: float = 2e3, connection: str = "A",
                       emf: float = 10.0, R_A: float = 0.5, R_V: float = 1e6,
                       accuracy_class: float = 0.5, counts: int = 2000,
                       r_source: float = 0.0, P_max: float | None = None,
                       closed: bool = True) -> Circuit:
    """Два резистора параллельно между a и 0. R_экв = R1·R2/(R1 + R2)."""
    load = [_resistor("R1", [NODE_A, GND], R1, P_max), _resistor("R2", [NODE_A, GND], R2, P_max)]
    return _assemble(load, connection, emf, R_A, R_V, accuracy_class, counts, r_source, closed)


def lamp_circuit(lamp_model: str = "6V_3W", connection: str = "A", emf: float = 10.0,
                 R_A: float = 0.5, R_V: float = 1e6, accuracy_class: float = 0.5,
                 counts: int = 2000, r_source: float = 0.0, closed: bool = True) -> Circuit:
    """Лампа накаливания вместо резистора — нелинейная ВАХ, закон Ома не выполняется.

    Модели ламп берутся из ядра (Lamp.presets). Лампа может перегореть
    (solve(auto_burn=True)), поэтому для каждого эксперимента цепь нужно собирать заново.
    """
    load = [Lamp("L", [NODE_A, GND], model=lamp_model)]
    return _assemble(load, connection, emf, R_A, R_V, accuracy_class, counts, r_source, closed)


# ---------------------------------------------------------------------------
# Реестр схем — по нему строится выбор в интерфейсе и скрипт эксперимента
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    builder: Callable[..., Circuit]
    load_params: tuple[str, ...]                    # параметры нагрузки (R, R1, R2, lamp_model)
    fixed_connection: str | None = None             # для схем А и Б способ подключения задан
    nominal: Callable[[dict], float | None] = field(default=lambda p: None)  # R_задан нагрузки
    linear: bool = True                             # омическая ли нагрузка

    def build(self, **params) -> Circuit:
        """Собрать цепь; лишние параметры (не нужные этой схеме) отбрасываются."""
        allowed = inspect.signature(self.builder).parameters
        kw = {k: v for k, v in params.items() if k in allowed}
        return self.builder(**kw)

    def connection(self, params: dict) -> str:
        return self.fixed_connection or params.get("connection", "A")


PRESETS: dict[str, Preset] = {
    p.key: p for p in (
        Preset("A", "Схема А — вольтметр параллельно резистору", scheme_a, ("R",),
               fixed_connection="A", nominal=lambda p: p["R"]),
        Preset("B", "Схема Б — вольтметр параллельно «амперметр + R»", scheme_b, ("R",),
               fixed_connection="B", nominal=lambda p: p["R"]),
        Preset("series", "Два резистора последовательно", series_resistors, ("R1", "R2"),
               nominal=lambda p: p["R1"] + p["R2"]),
        Preset("parallel", "Два резистора параллельно", parallel_resistors, ("R1", "R2"),
               nominal=lambda p: p["R1"] * p["R2"] / (p["R1"] + p["R2"])),
        Preset("lamp", "Лампа накаливания (нелинейный элемент)", lamp_circuit, ("lamp_model",),
               linear=False),
    )
}

LAMP_MODELS = tuple(Lamp.presets)


def build(key: str, **params) -> Circuit:
    """build("A", R=1e5, emf=10, R_V=1e6) — собрать готовую схему по ключу реестра."""
    try:
        preset = PRESETS[key]
    except KeyError:
        raise ValueError(f"Нет готовой схемы «{key}». Есть: {', '.join(PRESETS)}") from None
    return preset.build(**params)
