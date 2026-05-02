from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcgen-dir", default=None)
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    bindings_dir = root / "bindings"
    build_dir = root / "build" / "gcgen_pybind"
    gcgen_arg = args.gcgen_dir or os.environ.get("GCGEN_DIR")

    if gcgen_arg is None:
        raise RuntimeError("Set --gcgen-dir or GCGEN_DIR")

    gcgen_dir = Path(gcgen_arg).resolve()

    if not (gcgen_dir / "src" / "IConstrainedOptProblem.hpp").exists():
        raise RuntimeError(f"GCGen directory is incorrect: {gcgen_dir}")

    build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        str(bindings_dir),
        "-B",
        str(build_dir),
        f"-DGCGEN_DIR={gcgen_dir}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        args.config,
    ]

    subprocess.check_call(configure_cmd)
    subprocess.check_call(build_cmd)

    candidates = [
        p for p in build_dir.rglob("gcgen_py*")
        if p.is_file() and p.suffix.lower() in {".pyd", ".so", ".dylib"}
    ]

    if not candidates:
        raise RuntimeError("compiled gcgen_py module was not found")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    target = root / candidates[0].name
    shutil.copy2(candidates[0], target)

    print(f"built module: {target}")


if __name__ == "__main__":
    main()