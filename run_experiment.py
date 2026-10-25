#!/usr/bin/env python3
"""Entry point for the cross-task failure-memory experiment (see experiment_cli.py)."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reflexion_reproduction.experiment_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
