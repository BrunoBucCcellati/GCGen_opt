from __future__ import annotations

import math

from gcgen_adapter import make_problem_from_gcgen
from optimizer import SolverParameters, StronginMultistepOptimizer


def fmt_float(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.12g}"


def fmt_vector(values) -> str:
    if values is None:
        return "None"
    return "(" + ", ".join(fmt_float(v) for v in values) + ")"


def distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def default_solver_parameters(native) -> SolverParameters:
    dim = int(native.dimension())

    if dim <= 2:
        return SolverParameters(
            reliability=2.5,
            eps=1e-3,
            iterations_per_level=60,
            max_function_trials=50000,
            initial_lipschitz=1.0,
        )

    if dim == 3:
        return SolverParameters(
            reliability=2.5,
            eps=2e-3,
            iterations_per_level=35,
            max_function_trials=60000,
            initial_lipschitz=1.0,
        )

    return SolverParameters(
        reliability=2.5,
        eps=5e-3,
        iterations_per_level=20,
        max_function_trials=100000,
        initial_lipschitz=1.0,
    )


def is_feasible(constraints, tol: float = 1e-7) -> bool:
    return all(float(v) <= tol for v in constraints)


def in_bounds(point, low, high, tol: float = 1e-12) -> bool:
    return all(float(a) - tol <= float(x) <= float(b) + tol for x, a, b in zip(point, low, high))


def candidate_record(native, source, point):
    point = tuple(float(v) for v in point)
    low, high = native.bounds()
    constraints = tuple(float(v) for v in native.constraints(list(point)))
    value = float(native.objective(list(point)))

    return {
        "source": source,
        "point": point,
        "value": value,
        "constraints": constraints,
        "feasible": is_feasible(constraints),
        "in_bounds": in_bounds(point, low, high),
    }


def make_reference(native):
    target_value = float(native.optimum_value())
    reported_point = tuple(float(v) for v in native.optimum_point())
    candidates = []
    seen = set()

    def add(source, point):
        key = tuple(round(float(v), 12) for v in point)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate_record(native, source, point))

    add("GCGen GetOptimumPoint()", reported_point)

    if native.family().lower() == "optsq":
        add("OptSq reflected candidate", tuple(-0.5 * v for v in reported_point))
        add("zero point", tuple(0.0 for _ in reported_point))

    for c in candidates:
        c["value_error"] = abs(c["value"] - target_value)

    best = min(candidates, key=lambda c: (not c["in_bounds"], not c["feasible"], c["value_error"]))
    tol = max(1e-7, 1e-6 * (1.0 + abs(target_value)))

    if best["in_bounds"] and best["feasible"] and best["value_error"] <= tol:
        reference_point = best["point"]
        reference_source = best["source"]
        reference_value_at_point = best["value"]
        reference_constraints = best["constraints"]
    else:
        reference_point = None
        reference_source = None
        reference_value_at_point = None
        reference_constraints = None

    reported = candidates[0]

    return {
        "value": target_value,
        "point": reference_point,
        "source": reference_source,
        "point_value": reference_value_at_point,
        "constraints": reference_constraints,
        "reported_point": reported["point"],
        "reported_value": reported["value"],
        "reported_constraints": reported["constraints"],
        "reported_value_error": reported["value_error"],
        "reported_feasible": reported["feasible"],
    }


def print_parameters(params: SolverParameters) -> None:
    print(
        f"Параметры алгоритма: "
        f"r={params.reliability}, "
        f"eps={params.eps}, "
        f"iters_per_level={params.iterations_per_level}, "
        f"max_trials={params.max_function_trials}, "
        f"M0={params.initial_lipschitz}"
    )


def run_gcgen_problem(native, params: SolverParameters | None = None):
    params = params or default_solver_parameters(native)
    problem = make_problem_from_gcgen(native)
    reference = make_reference(native)
    low, high = native.bounds()

    print("\n" + "=" * 60)
    print("GCGEN ЗАДАЧА")
    print(f"Семейство: {native.family()}")
    print(f"Описание: {native.description()}")
    print(f"Размерность: {native.dimension()}")
    print(f"Ограничений: {native.constraints_number()}")
    print(f"Нижние границы: {fmt_vector(low)}")
    print(f"Верхние границы: {fmt_vector(high)}")
    print_parameters(params)
    print("-" * 60)
    print(f"GCGen GetOptimumPoint(): {fmt_vector(reference['reported_point'])}")
    print(f"F(GetOptimumPoint()): {fmt_float(reference['reported_value'])}")
    print(f"g_i(GetOptimumPoint()): {fmt_vector(reference['reported_constraints'])}")
    print(f"GetOptimumValue(): {fmt_float(reference['value'])}")
    print(f"Отклонение F(GetOptimumPoint()) от GetOptimumValue(): {reference['reported_value_error']:.2e}")

    if reference["point"] is not None:
        print("-" * 60)
        print(f"Эталонная точка для сравнения: {fmt_vector(reference['point'])}")
        print(f"Источник эталона: {reference['source']}")
        print(f"F(эталонной точки): {fmt_float(reference['point_value'])}")
        print(f"g_i(эталонной точки): {fmt_vector(reference['constraints'])}")
    else:
        print("-" * 60)
        print("Эталонная точка для сравнения: не выбрана")
        print("Сравнение будет выполнено только по значению функции")

    print("=" * 60)

    solver = StronginMultistepOptimizer(params)
    result = solver.solve(problem)

    best_constraints = tuple(float(v) for v in native.constraints(list(result.best_point)))
    max_constraint = max(best_constraints) if best_constraints else 0.0
    point_error = distance(result.best_point, reference["point"]) if reference["point"] is not None else None

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print(f"Лучшая точка: {fmt_vector(result.best_point)}")
    print(f"Лучшее значение: {fmt_float(result.best_value)}")
    print(f"Индекс: {result.best_index}")
    print(f"Допустима: {result.feasible}")
    print(f"Вызовов функции: {result.function_trials}")
    print(f"Время: {result.elapsed_time_sec:.4f} сек")
    print(f"Достигнутая точность: {result.achieved_accuracy}")
    print(f"Ограничения в найденной точке: {fmt_vector(best_constraints)}")
    print(f"max g_i(x): {fmt_float(max_constraint)}")
    print("-" * 60)
    print(f"Эталонное значение: {fmt_float(reference['value'])}")

    if reference["point"] is not None:
        print(f"Эталонная точка: {fmt_vector(reference['point'])}")

    if result.feasible:
        print(f"Разница по функции: {abs(float(result.best_value) - float(reference['value'])):.2e}")

    if point_error is not None:
        print(f"Расстояние до эталонной точки: {point_error:.2e}")

    print("=" * 60)
    return result