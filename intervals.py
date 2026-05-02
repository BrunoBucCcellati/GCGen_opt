from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from trial_types import Trial


@dataclass
class LipschitzBook:
    M: float = 1.0
    best_feasible_value: float = float("inf")
    revision: int = 0

    def z_value(self, trial: Trial) -> float:
        if trial.feasible and self.best_feasible_value != float("inf"):
            return trial.value - self.best_feasible_value
        return trial.value

    def observe_feasible(self, value: float) -> bool:
        if value < self.best_feasible_value:
            self.best_feasible_value = float(value)
            self.revision += 1
            return True
        return False

    def update_M(self, candidate: float) -> bool:
        if candidate > self.M:
            self.M = float(candidate)
            self.revision += 1
            return True
        return False

    def working_m(self, r: float) -> float:
        return r * self.M


class SearchInterval:
    def __init__(self, left: Trial, right: Trial):
        self.left = left
        self.right = right
        self.characteristic = 0.0

    @property
    def width(self) -> float:
        return self.right.level_coordinate - self.left.level_coordinate

    @property
    def a(self) -> float:
        return self.left.level_coordinate

    @property
    def b(self) -> float:
        return self.right.level_coordinate

    def local_lipschitz_candidate(self, lipschitz: LipschitzBook) -> float | None:
        if self.left.index != self.right.index:
            return None
        diff = abs(lipschitz.z_value(self.right) - lipschitz.z_value(self.left))
        return diff / self.width

    def recompute_characteristic(self, lipschitz: LipschitzBook, r: float) -> float:
        m = lipschitz.working_m(r)
        d = self.width
        z_l = lipschitz.z_value(self.left)
        z_r = lipschitz.z_value(self.right)

        if self.left.index == self.right.index:
            delta = z_r - z_l
            self.characteristic = (delta * delta) / (m * d) + m * d - 2.0 * (z_l + z_r)
        elif self.left.index < self.right.index:
            self.characteristic = 2.0 * m * d - 4.0 * z_r
        else:
            self.characteristic = 2.0 * m * d - 4.0 * z_l
        return self.characteristic

    def next_coordinate(self, lipschitz: LipschitzBook, r: float) -> float:
        m = lipschitz.working_m(r)
        mid = 0.5 * (self.a + self.b)

        if self.left.index == self.right.index:
            x_new = mid - (lipschitz.z_value(self.right) - lipschitz.z_value(self.left)) / (2.0 * m)
        elif self.left.index < self.right.index:
            x_new = mid - lipschitz.z_value(self.right) / (2.0 * m)
        else:
            x_new = mid + lipschitz.z_value(self.left) / (2.0 * m)

        if x_new <= self.a or x_new >= self.b:
            x_new = mid
        return x_new


class IntervalQueue:
    def __init__(self, lipschitz: LipschitzBook, r: float):
        self._heap = []
        self._counter = itertools.count()
        self._revision = lipschitz.revision
        self.lipschitz = lipschitz
        self._r = r

    def push(self, interval: SearchInterval) -> None:
        cand = interval.local_lipschitz_candidate(self.lipschitz)
        if cand is not None:
            self.lipschitz.update_M(cand)
        r_val = interval.recompute_characteristic(self.lipschitz, self._r)
        heapq.heappush(self._heap, (-r_val, next(self._counter), interval))

    def actualize(self) -> None:
        if self._revision != self.lipschitz.revision:
            new_heap = []
            for _, _, interval in self._heap:
                r_val = interval.recompute_characteristic(self.lipschitz, self._r)
                heapq.heappush(new_heap, (-r_val, next(self._counter), interval))
            self._heap = new_heap
            self._revision = self.lipschitz.revision

    def pop(self) -> SearchInterval:
        self.actualize()
        return heapq.heappop(self._heap)[2]

    def top_width(self) -> float:
        self.actualize()
        return self._heap[0][2].width
