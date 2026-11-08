"""
core_api.py — единственная точка, где мой код подключается к ядру напарника.

Ядро (solver.py, components.py) лежит в корне репозитория, а мой код — в src/.
Этот модуль добавляет корень репозитория в sys.path и реэкспортирует нужные
классы, чтобы остальные модули писали просто `from core_api import Circuit`.

Ядро здесь НЕ изменяется и НЕ дублируется.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from components import (  # noqa: E402  — импорт после настройки sys.path
    GROUND_NAMES,
    Ammeter,
    ElementError,
    Lamp,
    Resistor,
    Switch,
    VoltageSource,
    Voltmeter,
)
from solver import Circuit, CircuitError, Solution  # noqa: E402

# Ошибки ядра, которые показываем пользователю как есть (текст уже на русском):
#   ElementError — неверные параметры элемента (бросается при сборке цепи);
#   CircuitError — цепь не решается (нет земли, КЗ, расчёт не сошёлся).
CORE_ERRORS = (ElementError, CircuitError)

__all__ = [
    "ROOT", "GROUND_NAMES", "CORE_ERRORS",
    "Ammeter", "Voltmeter", "Resistor", "Lamp", "Switch", "VoltageSource",
    "Circuit", "CircuitError", "ElementError", "Solution",
]
