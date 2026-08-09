"""Isolated Fintech Token research entry point.

The substantive implementation lives under ``research/fintech_token/`` so the
shared research notebook and notes remain untouched.  This thin module keeps
the requested ``research/fintech_research.py`` entry point available for
rerunning the dedicated audit.
"""

from pathlib import Path
import sys


# Make ``python research/fintech_research.py`` work from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.fintech_token.fintech_models import *  # noqa: F401,F403
from research.fintech_token.fintech_validation import run_all


if __name__ == "__main__":
    run_all()
