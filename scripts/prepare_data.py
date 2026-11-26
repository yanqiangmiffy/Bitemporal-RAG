#!/usr/bin/env python
"""raw datasets/ -> canonical benchmark. See `asof-prepare --help`."""
import _bootstrap  # noqa: F401

from asof.cli import prepare_data_main

if __name__ == "__main__":
    prepare_data_main()
