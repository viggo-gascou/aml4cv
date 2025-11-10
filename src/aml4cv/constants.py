"""Constants used throughout the project."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

RESULTS_DIR = BASE_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()
