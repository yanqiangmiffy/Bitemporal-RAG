#!/usr/bin/env python
"""Results json -> paper LaTeX tables. See `asof-tables --help`."""
import _bootstrap  # noqa: F401

from asof.cli import make_tables_main

if __name__ == "__main__":
    make_tables_main()
