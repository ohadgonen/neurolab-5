"""MAT-file loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import scipy.io as sio


def load_mat(path: str | Path) -> Dict[str, Any]:
    """Load a MATLAB .mat file with consistent options."""
    return sio.loadmat(Path(path), squeeze_me=True, struct_as_record=False)
