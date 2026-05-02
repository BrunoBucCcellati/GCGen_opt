from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from trial_types import Trial

Vector = list[float]


@dataclass(frozen=True)
class FloatParameter:
    name: str
    low: float
    high: float


@dataclass(frozen=True)
class Constraint:
    name: str
    func: Callable[[Sequence[float]], float]


@dataclass
class OptimizationProblem:
    parameters: list[FloatParameter] = field(default_factory=list)
    objective: Callable[[Sequence[float]], float] | None = None
    constraints: list[Constraint] = field(default_factory=list)

    def add_float_parameter(self, name: str, low: float, high: float) -> FloatParameter:
        p = FloatParameter(name=name, low=float(low), high=float(high))
        self.parameters.append(p)
        return p

    def add_constraint(self, name: str, func: Callable[[Sequence[float]], float]) -> Constraint:
        c = Constraint(name=name, func=func)
        self.constraints.append(c)
        return c

    def set_objective(self, func: Callable[[Sequence[float]], float]) -> None:
        self.objective = func

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def objective_index(self) -> int:
        return len(self.constraints) + 1

    def bounds_for(self, level: int) -> tuple[float, float]:
        p = self.parameters[level]
        return p.low, p.high

    def evaluate_indexed(self, x: Sequence[float]) -> Trial:
        values: list[float] = []
        for i, constraint in enumerate(self.constraints, start=1):
            val = float(constraint.func(x))
            values.append(val)
            if val > 0.0:
                return Trial(
                    x=tuple(x),
                    index=i,
                    value=val,
                    feasible=False,
                )
        return Trial(
            x=tuple(x),
            index=self.objective_index,
            value=float(self.objective(x)),
            feasible=True,
        )
