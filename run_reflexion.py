#!/usr/bin/env python3
"""Convenient entry point for the Reflexion reproduction."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reflexion_reproduction.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

