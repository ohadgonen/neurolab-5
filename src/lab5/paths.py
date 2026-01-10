"""paths for the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "matlab results"
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_FINAL_DIR = PLOTS_DIR / "final"
