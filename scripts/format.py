#!/usr/bin/env python3
"""Format and lint-fix codebase with a single command.

Usage:
    python scripts/format.py           # Auto-format and auto-fix in place
    python scripts/format.py --check   # Check formatting and lint without modifying files
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_PATHS = ["src", "tests", "scripts"]


def run_command(cmd: list[str], cwd: Path) -> int:
    """Run a command using venv ruff, PATH ruff, uv run ruff, or python -m ruff."""
    # 1. Check local virtualenv
    for venv_candidate in [cwd / ".venv" / "bin" / "ruff", cwd / ".venv" / "Scripts" / "ruff.exe"]:
        if venv_candidate.exists():
            res = subprocess.run([str(venv_candidate), *cmd[1:]], cwd=cwd, check=False)
            return res.returncode

    # 2. Try ruff in system PATH
    try:
        res = subprocess.run(["ruff", *cmd[1:]], cwd=cwd, check=False)
        return res.returncode
    except FileNotFoundError:
        pass

    # 3. Try uv run ruff
    try:
        res = subprocess.run(
            ["uv", "run", "--extra", "dev", "ruff", *cmd[1:]],
            cwd=cwd,
            check=False,
        )
        return res.returncode
    except FileNotFoundError:
        pass

    # 4. Fallback to current python -m ruff
    res = subprocess.run([sys.executable, "-m", "ruff", *cmd[1:]], cwd=cwd, check=False)
    return res.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run code formatter and autofix linters on the neural-cost codebase."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check formatting and linting without modifying files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Paths to format (defaults to src, tests, scripts, benchmarks, examples).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    target_paths = [p for p in args.paths if (repo_root / p).exists()]

    if not target_paths:
        print("No valid paths found to format.")
        return 0

    if args.check:
        print(f"Checking formatting with ruff format --check on {target_paths}...")
        res_fmt = run_command(["ruff", "format", "--check", *target_paths], cwd=repo_root)
        if res_fmt != 0:
            print("Formatting check failed! Run 'python scripts/format.py' to fix.")
            return res_fmt

        print(f"Checking lint with ruff check on {target_paths}...")
        res_lint = run_command(["ruff", "check", *target_paths], cwd=repo_root)
        return res_lint

    print(f"Formatting code with ruff format on {target_paths}...")
    res_fmt = run_command(["ruff", "format", *target_paths], cwd=repo_root)
    if res_fmt != 0:
        return res_fmt

    print(f"Fixing lint and import sorting with ruff check --fix on {target_paths}...")
    res_lint = run_command(["ruff", "check", "--fix", *target_paths], cwd=repo_root)
    if res_lint == 0:
        print("Formatting and lint fixes completed successfully!")
    return res_lint


if __name__ == "__main__":
    sys.exit(main())
