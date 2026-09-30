"""Run the whole Astra test suite: ``python -m tests``.

Runs the unittest-based tests plus the two script-style suites that predate
them (store internals and end-to-end persistence).
"""
from __future__ import annotations

import sys
import unittest


def main() -> int:
    suite = unittest.TestLoader().discover(start_dir="tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    from tests import test_memory_store, test_persistence

    test_memory_store.run_all()
    test_persistence.run_tests()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
