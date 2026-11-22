"""Pytest bootstrap: make the repository root importable.

The tests live in ``tests/`` without an ``__init__.py``; adding the repository
root to ``sys.path`` lets ``import deferred`` work from the default pytest
import mode.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
