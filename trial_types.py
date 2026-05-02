from dataclasses import dataclass

@dataclass
class Trial:
    x: tuple[float, ...]
    index: int
    value: float
    feasible: bool
    constraint_values: tuple[float, ...]
    level_coordinate: float | None = None