from __future__ import annotations

from gcgen_adapter import create_native_gcgen_problem
from gcgen_runner import run_gcgen_problem


def main() -> None:
    cases = [
        (
            "GKLS: минимум внутри допустимой области",
            dict(
                family="gkls",
                problem_type="in",
                fraction=0.3,
                active_constraints=0,
                problem_index=1,
                dim=2,
                gkls_class="Simple",
                gkls_function_type="TD",
            ),
        ),
        (
            "GKLS: минимум вне допустимой области",
            dict(
                family="gkls",
                problem_type="out",
                fraction=0.5,
                active_constraints=0,
                problem_index=2,
                dim=2,
                gkls_class="Simple",
                gkls_function_type="TD",
            ),
        ),
        (
            "GKLS: минимум на границе допустимой области",
            dict(
                family="gkls",
                problem_type="border",
                fraction=0.5,
                active_constraints=1,
                problem_index=3,
                dim=2,
                gkls_class="Simple",
                gkls_function_type="TD",
            ),
        ),
        (
            "Grishagin: минимум внутри допустимой области",
            dict(
                family="grishagin",
                problem_type="in",
                fraction=0.3,
                active_constraints=0,
                problem_index=1,
            ),
        ),
        (
            "Grishagin: минимум вне допустимой области",
            dict(
                family="grishagin",
                problem_type="out",
                fraction=0.5,
                active_constraints=0,
                problem_index=2,
            ),
        ),
        (
            "OptSq: нормальная квадратичная задача",
            dict(
                family="optsq",
                problem_type="normal",
                fraction=1.0,
                active_constraints=0,
                dim=2,
                low=-10.0,
                high=10.0,
            ),
        ),
        (
            "OptSq: квадратичная задача с преобразованием border",
            dict(
                family="optsq",
                problem_type="border",
                fraction=0.5,
                active_constraints=1,
                dim=2,
                low=-10.0,
                high=10.0,
            ),
        ),
    ]

    for title, cfg in cases:
        print("\n\n" + "#" * 60)
        print(title)
        print("#" * 60)

        native = create_native_gcgen_problem(**cfg)
        run_gcgen_problem(native)


if __name__ == "__main__":
    main()