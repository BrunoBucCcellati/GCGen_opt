from __future__ import annotations

import time
from dataclasses import dataclass

from intervals import IntervalQueue, LipschitzBook, SearchInterval
from problem import OptimizationProblem
from trial_types import Trial


def is_better_indexed(a: Trial | None, b: Trial | None) -> bool:
    if b is None:
        return True
    if a.index != b.index:
        return a.index > b.index
    return a.value < b.value


class SearchStatistics:
    def __init__(self, problem: OptimizationProblem, max_trials: int):
        self.problem = problem
        self.max_trials = max_trials
        self.trials = 0
        self.best_feasible: Trial | None = None
        self.best_indexed: Trial | None = None
        self.start = time.perf_counter()

    def can_evaluate(self) -> bool:
        return self.trials < self.max_trials

    def evaluate(self, x: tuple[float, ...], level_coordinate: float) -> Trial:
        trial = self.problem.evaluate_indexed(x)
        trial.level_coordinate = level_coordinate
        self.trials += 1
        if is_better_indexed(trial, self.best_indexed):
            self.best_indexed = trial
        if trial.feasible:
            if self.best_feasible is None or trial.value < self.best_feasible.value:
                self.best_feasible = trial
        return trial

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@dataclass
class SolverResult:
    best_point: tuple[float, ...]
    best_value: float
    best_index: int
    feasible: bool
    function_trials: int
    elapsed_time_sec: float
    achieved_accuracy: float


@dataclass(frozen=True)
class SolverParameters:
    reliability: float = 2.5
    eps: float = 1e-3
    iterations_per_level: int = 25
    max_function_trials: int = 10000
    initial_lipschitz: float = 1.0


class StronginMultistepOptimizer:
    def __init__(self, parameters: SolverParameters | None = None) -> None:
        self.par = parameters or SolverParameters()
        self._stats = None
        self._lipschitz = None

    def solve(self, problem: OptimizationProblem) -> SolverResult:
        self._last_accuracy = float("inf")
        self._stats = SearchStatistics(problem, self.par.max_function_trials)
        self._lipschitz = LipschitzBook(reliability=self.par.reliability, M=self.par.initial_lipschitz)
        self._optimize_level(0, ())
        best = self._stats.best_feasible or self._stats.best_indexed
        return SolverResult(
            best_point=best.x,
            best_value=best.value,
            best_index=best.index,
            feasible=best.feasible,
            function_trials=self._stats.trials,
            elapsed_time_sec=self._stats.elapsed(),
            achieved_accuracy=self._last_accuracy,
        )

    def _optimize_level(self, level: int, prefix: tuple[float, ...]) -> Trial:
        low, high = self._stats.problem.bounds_for(level)
        queue = IntervalQueue(self._lipschitz)
        left = self._evaluate_level_point(level, prefix, low)
        right = self._evaluate_level_point(level, prefix, high)
        best = left if is_better_indexed(left, right) else right
        if high - low <= self.par.eps:
            self._last_accuracy = min(self._last_accuracy, high - low)
            return best
        queue.push(SearchInterval(left, right))
        for _ in range(self.par.iterations_per_level):
            if not self._stats.can_evaluate():
                break
            interval = queue.pop()
            if interval.width <= self.par.eps:
                self._last_accuracy = min(self._last_accuracy, interval.width)
                break
            x_new = interval.next_coordinate(self._lipschitz)
            trial = self._evaluate_level_point(level, prefix, x_new)
            if is_better_indexed(trial, best):
                best = trial
            queue.push(SearchInterval(interval.left, trial))
            queue.push(SearchInterval(trial, interval.right))
            self._last_accuracy = min(self._last_accuracy, interval.width)
        return best

    def _evaluate_level_point(self, level: int, prefix: tuple[float, ...], coord: float) -> Trial:
        low, high = self._stats.problem.bounds_for(level)
        coord = min(max(coord, low), high)
        new_prefix = prefix + (coord,)
        if level == self._stats.problem.dimension - 1:
            trial = self._stats.evaluate(new_prefix, coord)
        else:
            inner = self._optimize_level(level + 1, new_prefix)
            trial = Trial(
                x=inner.x,
                index=inner.index,
                value=inner.value,
                feasible=inner.feasible,
                level_coordinate=coord,
            )
            if is_better_indexed(trial, self._stats.best_indexed):
                self._stats.best_indexed = trial
        if trial.feasible:
            self._lipschitz.observe_feasible(trial.value)
        return trial
