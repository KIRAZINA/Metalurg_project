"""Backward-compatible CLI entry point.

The implementation lives in :mod:`test_metal.cli`; this shim keeps the
historical ``python main.py ...`` invocation working.
"""

from test_metal.cli import configure_logging, main

__all__ = ["configure_logging", "main"]


if __name__ == "__main__":
    main()
