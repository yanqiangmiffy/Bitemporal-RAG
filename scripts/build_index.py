#!/usr/bin/env python
"""Build dense+sparse index (GPU step). See `asof-build-index --help`."""
import _bootstrap  # noqa: F401

from asof.cli import build_index_main

if __name__ == "__main__":
    build_index_main()
