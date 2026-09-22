#!/usr/bin/env python3
"""Debug GUI launcher:  ./debug_gui.py     (or:  python3 debug_gui.py)

The normal GUI plus the switches that bring the notebook bugs back, to A/B
them against the fix.  For real runs use ``./gui.py``.
"""

import sys

from gui import launch

if __name__ == "__main__":
    sys.exit(launch("quantum_rl.debug_gui", __file__))
