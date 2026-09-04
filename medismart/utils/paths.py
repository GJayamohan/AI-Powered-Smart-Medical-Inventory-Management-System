"""Central location for every path in the project.

Nothing else should hardcode a directory. Import from here so the whole
project keeps working if the tree is moved or renamed.
"""

from __future__ import annotations

from pathlib import Path

# medismart/utils/paths.py -> medismart/utils -> medismart -> project root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
KNOWLEDGE = DATA / "knowledge"

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
REPORTS = ARTIFACTS / "reports"

DOCS = ROOT / "docs"
SCRIPTS = ROOT / "scripts"

_ALL_DIRS = [DATA, RAW, PROCESSED, KNOWLEDGE, ARTIFACTS, MODELS, FIGURES, REPORTS]


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
