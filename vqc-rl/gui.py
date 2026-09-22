#!/usr/bin/env python3
"""GUI launcher:  ./gui.py     (or:  python3 gui.py)

Always trains the corrected agent.  To re-enable the notebook bugs use
``./debug_gui.py`` instead.

Automatically uses the venv interpreter -- the one that has pennylane and torch --
so there is no long path to remember.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")
VENV = os.path.join(VENV_DIR, "bin", "python")


def _inside_venv() -> bool:
    """Are we already running inside .venv?

    The comparison must be made on ``sys.prefix``, NOT on the executable path:
    ``.venv/bin/python`` is a symlink to the system interpreter, so two
    ``realpath`` calls would match even when the venv is not active, and the
    relaunch would never happen.
    """
    return os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR)


def launch(module: str, script: str) -> int:
    """Run ``<module>.main()`` with the venv interpreter.

    ``script`` is the launcher being run: it is what gets re-executed inside the
    venv, so ``debug_gui.py`` relaunches itself and not this file.
    """
    # if we are not inside the venv yet, relaunch with its interpreter
    if os.path.exists(VENV) and not _inside_venv():
        return subprocess.call([VENV, os.path.abspath(script)] + sys.argv[1:])

    sys.path.insert(0, HERE)
    try:
        gui_main = __import__(module, fromlist=["main"]).main
    except ImportError as e:
        missing = getattr(e, "name", "") or str(e)
        print(f"Missing dependency: {missing}", file=sys.stderr)
        if "tkinter" in str(e):
            print("Install Tk with:  sudo apt install python3-tk", file=sys.stderr)
        elif not os.path.exists(VENV):
            print("No .venv found. Create it with:", file=sys.stderr)
            print(f"  python3 -m venv {VENV_DIR}", file=sys.stderr)
            print(f"  {VENV} -m pip install -r {os.path.join(HERE, 'requirements.txt')}",
                  file=sys.stderr)
        else:
            print(f"Install with:  {VENV} -m pip install {missing}", file=sys.stderr)
        return 1

    gui_main()
    return 0


def main() -> int:
    return launch("quantum_rl.gui", __file__)


if __name__ == "__main__":
    sys.exit(main())
