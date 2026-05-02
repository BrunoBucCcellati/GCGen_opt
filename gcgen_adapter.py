from __future__ import annotations

from problem import OptimizationProblem

try:
    import gcgen_py
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "gcgen_py is not built. Run: python build_gcgen_binding.py --gcgen-dir PATH_TO_GCGEN"
    ) from exc


def create_native_gcgen_problem(
    family: str = "gkls",
    problem_type: str = "in",
    fraction: float = 0.5,
    active_constraints: int = 0,
    problem_index: int = 1,
    dim: int = 2,
    gkls_class: str = "Simple",
    gkls_function_type: str = "TD",
    low: float = -10.0,
    high: float = 10.0,
):
    name = family.lower().replace("_", "").replace("-", "")

    if name == "gkls":
        return gcgen_py.make_gkls(
            problem_type=problem_type,
            fraction=float(fraction),
            active_constraints=int(active_constraints),
            problem_index=int(problem_index),
            dim=int(dim),
            gkls_class=gkls_class,
            function_type=gkls_function_type,
        )

    if name == "grishagin":
        return gcgen_py.make_grishagin(
            problem_type=problem_type,
            fraction=float(fraction),
            active_constraints=int(active_constraints),
            problem_index=int(problem_index),
        )

    if name in {"optsq", "sq", "square"}:
        return gcgen_py.make_optsq(
            dim=int(dim),
            low=float(low),
            high=float(high),
            problem_type=problem_type,
            fraction=float(fraction),
            active_constraints=int(active_constraints),
        )

    raise ValueError(f"unknown GCGen family: {family}")


def make_problem_from_gcgen(native) -> OptimizationProblem:
    problem = OptimizationProblem()
    low, high = native.bounds()

    for i in range(native.dimension()):
        problem.add_float_parameter(f"x{i}", low[i], high[i])

    def objective(x):
        return native.objective(list(x))

    problem.set_objective(objective)

    for i in range(native.constraints_number()):
        def constraint(x, index=i):
            return native.constraint(index, list(x))

        problem.add_constraint(f"g{i + 1}", constraint)

    problem.gcgen_native = native
    return problem