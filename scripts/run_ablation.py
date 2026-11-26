#!/usr/bin/env python
"""Run all ablation variants on a domain. See `asof-ablation --help`."""
import _bootstrap  # noqa: F401

from asof.cli import run_ablation_main

if __name__ == "__main__":
    run_ablation_main()
