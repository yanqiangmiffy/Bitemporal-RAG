#!/usr/bin/env python
"""Run one method on one domain. See `asof-run --help`."""
import _bootstrap  # noqa: F401

from asof.cli import run_experiment_main

if __name__ == "__main__":
    run_experiment_main()
