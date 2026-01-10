"""Figure helpers for consistent styling and saving."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt

from .paths import PLOTS_FINAL_DIR
from .plot_style import RC_PARAMS


def apply_style() -> None:
    """Apply shared matplotlib rcParams once per session."""
    plt.rcParams.update(RC_PARAMS)


def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    out_dir: Path | None = None,
    fmt: Sequence[str] = ("png",),
    dpi: int = 300,
    tight: bool = True,
) -> Tuple[Path, ...]:
    """Save a figure to the output directory and return saved paths."""
    target_dir = Path(out_dir) if out_dir is not None else PLOTS_FINAL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for ext in fmt:
        path = target_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None)
        paths.append(path)
    return tuple(paths)


def set_shared_limits(
    axes: Iterable[plt.Axes],
    *,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    share: str = "both",
) -> None:
    """Set shared axis limits across a list of axes."""
    for ax in axes:
        if share in {"both", "x"} and xlim is not None:
            ax.set_xlim(xlim)
        if share in {"both", "y"} and ylim is not None:
            ax.set_ylim(ylim)


def compute_global_ylim(curves: Iterable[Sequence[float]], padding: float = 0.05) -> Tuple[float, float]:
    """Compute global y-limits for multiple curves with padding."""
    ymin = min(min(curve) for curve in curves)
    ymax = max(max(curve) for curve in curves)
    if ymax == ymin:
        return ymin - 1.0, ymax + 1.0
    pad = (ymax - ymin) * padding
    return ymin - pad, ymax + pad
