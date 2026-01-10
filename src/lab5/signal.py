"""Signal-level plotting helpers (Part 1)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from .constants import FILTER_ORDER, FS_SPIKE, HP_CUTOFF_HZ, LP_CUTOFF_HZ
from .plot_style import COLORS, FIGSIZE_WIDE, LABELS


def plotSignal(
    raw_signal: np.ndarray,
    t_start: float = 0.0,
    t_duration: float = 0.1,
    *,
    fs: float = FS_SPIKE,
    hp_cutoff: float = HP_CUTOFF_HZ,
    lp_cutoff: float = LP_CUTOFF_HZ,
    order: int = FILTER_ORDER,
    scale: float = 1e3,
    show_stats: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot raw, low-pass, and high-pass filtered signal segments."""
    raw_signal = np.asarray(raw_signal, dtype=float).squeeze()
    raw_signal = raw_signal * scale

    if show_stats:
        stats = (
            f"n={raw_signal.size}, mean={raw_signal.mean():.3g}, "
            f"std={raw_signal.std():.3g}, min={raw_signal.min():.3g}, "
            f"max={raw_signal.max():.3g}"
        )
        print(f"raw_signal: {stats}")

    start_idx = int(round(t_start * fs))
    end_idx = int(round((t_start + t_duration) * fs))
    end_idx = min(end_idx, raw_signal.size)

    sig = raw_signal[start_idx:end_idx]
    t = (np.arange(sig.size) / fs) + t_start

    nyq = fs / 2.0
    b_lp, a_lp = butter(order, lp_cutoff / nyq, btype="low")
    sig_low = filtfilt(b_lp, a_lp, sig)

    b_hp, a_hp = butter(order, hp_cutoff / nyq, btype="high")
    sig_high = filtfilt(b_hp, a_hp, sig)

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_WIDE, sharex=True)
    axes[0].plot(t, sig, color=COLORS["raw"])
    axes[0].set_title("Raw signal")
    axes[0].set_ylabel(LABELS["amplitude_uv"])

    axes[1].plot(t, sig_low, color=COLORS["lowpass"])
    axes[1].set_title("Low-pass filtered signal (LFP)")
    axes[1].set_ylabel(LABELS["amplitude_uv"])

    axes[2].plot(t, sig_high, color=COLORS["highpass"])
    axes[2].set_title("High-pass filtered signal (Spikes)")
    axes[2].set_ylabel(LABELS["amplitude_uv"])
    axes[2].set_xlabel(LABELS["time_s"])

    fig.tight_layout()
    return fig, axes
