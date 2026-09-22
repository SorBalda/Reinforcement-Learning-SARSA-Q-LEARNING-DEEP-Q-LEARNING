"""Make ``import quantum_rl`` work no matter how pytest is invoked.

There is no packaging metadata in this repo, so the repository root has to be on
``sys.path``.  ``python -m pytest`` from the root gets that for free; a bare
``pytest tests/`` does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    # `str(HERE)` lets the tests import `notebook_cell50_reference`, the literal
    # transcription of notebook cell 50 used by the differential test.
    if path not in sys.path:
        sys.path.insert(0, path)
