from dataclasses import dataclass

@dataclass
class Trial:
    x: tuple[float, ...]
    index: int
    value: float
    feasible: bool
    level_coordinate: float | None = None
