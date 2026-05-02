from __future__ import annotations

import argparse

from gcgen_adapter import create_native_gcgen_problem
from gcgen_runner import run_gcgen_problem


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--family", default="gkls")
    parser.add_argument("--problem-type", "--type", dest="problem_type", default="in")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--active-constraints", type=int, default=0)
    parser.add_argument("--problem-index", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--gkls-class", default="Simple")
    parser.add_argument("--gkls-function-type", default="TD")
    parser.add_argument("--low", type=float, default=-10.0)
    parser.add_argument("--high", type=float, default=10.0)

    args = parser.parse_args()

    native = create_native_gcgen_problem(
        family=args.family,
        problem_type=args.problem_type,
        fraction=args.fraction,
        active_constraints=args.active_constraints,
        problem_index=args.problem_index,
        dim=args.dim,
        gkls_class=args.gkls_class,
        gkls_function_type=args.gkls_function_type,
        low=args.low,
        high=args.high,
    )

    run_gcgen_problem(native)


if __name__ == "__main__":
    main()