"""
solver.py — расчёт цепи постоянного тока модифицированным узловым методом (MNA).

Неизвестные: потенциалы всех узлов (кроме земли) и токи ветвей
источников и амперметров. Уравнения: первый закон Кирхгофа в каждом узле
плюс уравнения ветвей. Нелинейные элементы (лампа, пользовательский элемент) линеаризуются,
система решается итерациями Ньютона до сходимости.

Чтобы «висящие» узлы не делали матрицу вырожденной, каждый узел соединён
с землёй очень малой проводимостью G_MIN (как в SPICE).
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from components import GROUND_NAMES, Element, ElementError, Lamp

G_MIN = 1e-12
REL_TOL = 1e-6
ABS_TOL = 1e-12


class CircuitError(Exception):
    """Цепь собрана неверно или не решается."""


# ---------------------------------------------------------------------------
# Сборка системы уравнений
# ---------------------------------------------------------------------------

class _System:
    """Матрица G и правая часть b для одной итерации: G·x = b."""

    def __init__(self, node_index: dict[str, int], size: int, x: np.ndarray):
        self.idx = node_index
        self.G = np.zeros((size, size))
        self.b = np.zeros(size)
        self.x = x
        self.limited = False

    def _i(self, node: str) -> int | None:
        return None if node in GROUND_NAMES else self.idx[node]

    def v(self, node: str) -> float:
        i = self._i(node)
        return 0.0 if i is None else float(self.x[i])

    def conductance(self, a: str, b: str, g: float) -> None:
        ia, ib = self._i(a), self._i(b)
        if ia is not None:
            self.G[ia, ia] += g
        if ib is not None:
            self.G[ib, ib] += g
        if ia is not None and ib is not None:
            self.G[ia, ib] -= g
            self.G[ib, ia] -= g

    def current(self, a: str, b: str, I: float) -> None:
        """Ток I течёт через элемент от узла a к узлу b."""
        ia, ib = self._i(a), self._i(b)
        if ia is not None:
            self.b[ia] -= I
        if ib is not None:
            self.b[ib] += I

    def branch(self, p: str, n: str, k: int, r: float, emf: float) -> None:
        """Ветвь с током i (втекает в p, вытекает из n): v_p − v_n − r·i = emf."""
        ip, in_ = self._i(p), self._i(n)
        if ip is not None:
            self.G[ip, k] += 1.0
            self.G[k, ip] += 1.0
        if in_ is not None:
            self.G[in_, k] -= 1.0
            self.G[k, in_] -= 1.0
        self.G[k, k] -= r
        self.b[k] += emf

    def linearized(self, node: str, coeffs: dict[str, float], const: float) -> None:
        """Ток, вытекающий из узла node в элемент: Σ coeffs[j]·v_j + const."""
        row = self._i(node)
        if row is None:
            return
        for n, c in coeffs.items():
            col = self._i(n)
            if col is not None:
                self.G[row, col] += c
        self.b[row] -= const


# ---------------------------------------------------------------------------
# Результат
# ---------------------------------------------------------------------------

class Solution:
    def __init__(self, circuit: "Circuit", x: np.ndarray, node_index: dict[str, int], iterations: int):
        self.circuit = circuit
        self.warnings: list[str] = []   # предупреждения проверки цепи (висящие узлы и т.п.)
        self.x = x
        self.node_index = node_index
        self.iterations = iterations

    def V(self, node: str) -> float:
        """Потенциал узла относительно земли."""
        if node in GROUND_NAMES:
            return 0.0
        try:
            return float(self.x[self.node_index[node]])
        except KeyError:
            raise CircuitError(f"Нет узла «{node}»") from None

    def report(self, name: str, noise: bool = False, rng: random.Random | None = None) -> dict:
        return self.circuit[name].report(self, rng, noise)

    def reports(self, noise: bool = False, seed: int | None = None) -> dict[str, dict]:
        """Отчёт по всем элементам. Для приборов — показания (с шумом, если noise=True)."""
        rng = random.Random(seed)
        return {name: el.report(self, rng, noise) for name, el in self.circuit.elements.items()}

    def node_voltages(self) -> dict[str, float]:
        return {n: self.V(n) for n in self.node_index}


# ---------------------------------------------------------------------------
# Цепь
# ---------------------------------------------------------------------------

class Circuit:
    def __init__(self, elements: Iterable[Element] = ()):
        self.elements: dict[str, Element] = {}
        for el in elements:
            self.add(el)

    # --- редактирование ----------------------------------------------------
    def add(self, el: Element) -> Element:
        if el.name in self.elements:
            raise CircuitError(f"Элемент с именем «{el.name}» уже есть")
        self.elements[el.name] = el
        return el

    def remove(self, name: str) -> None:
        self.elements.pop(name)

    def __getitem__(self, name: str) -> Element:
        try:
            return self.elements[name]
        except KeyError:
            raise CircuitError(f"Нет элемента «{name}»") from None

    def __contains__(self, name: str) -> bool:
        return name in self.elements

    @property
    def nodes(self) -> list[str]:
        seen: dict[str, None] = {}
        for el in self.elements.values():
            for n in el.nodes:
                if n not in GROUND_NAMES:
                    seen.setdefault(n)
        return list(seen)

    def to_specs(self) -> list[dict]:
        """Цепь в виде списка словарей (для сохранения в JSON)."""
        return [el.to_spec() for el in self.elements.values()]

    # --- проверка ----------------------------------------------------------
    def check(self) -> list[str]:
        """Ошибки бросает исключением, на возможные проблемы возвращает предупреждения."""
        if not self.elements:
            raise CircuitError("Цепь пустая")
        all_nodes = [n for el in self.elements.values() for n in el.nodes]
        if not any(n in GROUND_NAMES for n in all_nodes):
            raise CircuitError(f"Нет узла земли — назовите один из узлов {sorted(GROUND_NAMES)}")
        warnings = []
        for node in self.nodes:
            if all_nodes.count(node) < 2:
                owner = next(e.name for e in self.elements.values() if node in e.nodes)
                warnings.append(f"Узел «{node}» подключён только к {owner} — вывод висит в воздухе")
        for el in self.elements.values():
            warnings.extend(getattr(el, "warnings", []))
            if len(set(el.nodes)) < len(el.nodes):
                warnings.append(f"{el.name}: выводы замкнуты сами на себя")
        return warnings

    # --- расчёт ------------------------------------------------------------
    def solve(self, x0: np.ndarray | None = None, max_iter: int = 300,
              auto_burn: bool = False) -> Solution:
        """
        Рассчитать цепь.
          x0        — начальное приближение (например, решение предыдущей точки развёртки);
          auto_burn — если лампа превысила допустимую мощность, она перегорает
                      (размыкается) и цепь пересчитывается.
        """
        warnings = self.check()
        sol = self._solve_once(x0, max_iter)
        if auto_burn:
            for _ in range(len(self.elements)):
                burning = [el for el in self.elements.values()
                           if isinstance(el, Lamp) and el.report(sol)["will_burn"]]
                if not burning:
                    break
                for lamp in burning:
                    lamp.burned = True
                sol = self._solve_once(sol.x, max_iter)
        sol.warnings = warnings
        return sol

    def _solve_once(self, x0, max_iter) -> Solution:
        node_index = {n: i for i, n in enumerate(self.nodes)}
        n_nodes = len(node_index)
        k = n_nodes
        for el in self.elements.values():
            el.branch_index = k if el.n_branches else None
            k += el.n_branches
        size = k

        x = np.zeros(size) if x0 is None or len(x0) != size else np.array(x0, dtype=float)
        nonlinear = any(el.nonlinear for el in self.elements.values())

        start = _System(node_index, size, x)
        for el in self.elements.values():
            el.reset(start)

        for it in range(1, max_iter + 1):
            sys = _System(node_index, size, x)
            for el in self.elements.values():
                try:
                    el.stamp(sys)
                except (OverflowError, ValueError, ElementError) as exc:
                    raise CircuitError(f"{el.name}: ошибка расчёта ({exc})") from exc
            sys.G[np.arange(n_nodes), np.arange(n_nodes)] += G_MIN

            try:
                x_new = np.linalg.solve(sys.G, sys.b)
            except np.linalg.LinAlgError:
                raise CircuitError(self._singular_hint()) from None
            if not np.all(np.isfinite(x_new)) or not np.allclose(sys.G @ x_new, sys.b, rtol=1e-6, atol=1e-9):
                raise CircuitError(self._singular_hint())

            if not nonlinear:
                return Solution(self, x_new, node_index, it)
            if not sys.limited and np.all(np.abs(x_new - x) <= REL_TOL * np.abs(x_new) + ABS_TOL):
                return Solution(self, x_new, node_index, it)
            x = x_new

        raise CircuitError(f"Расчёт не сошёлся за {max_iter} итераций — проверьте параметры элементов")

    @staticmethod
    def _singular_hint() -> str:
        return ("Система уравнений не решается. Обычно это контур только из идеальных "
                "источников (r = 0) и амперметров (R_A = 0) — короткое замыкание источника, "
                "или два идеальных источника с разной ЭДС параллельно.")