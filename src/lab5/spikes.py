"""Spike waveform helpers (Part 2)."""

from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .constants import FS_SPIKE
from .plot_style import ALPHA_LIGHT, COLORS, LABELS, LINEWIDTH_THICK, LINEWIDTH_THIN


def ensure_waveforms_2d(
    waveforms: np.ndarray,
    *,
    samples_first: bool | None = None,
    n_spikes_expected: int | None = None,
    min_samples: int = 10,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, dict]:
    """Normalize waveform matrix to shape (n_samples, n_spikes)."""
    wf = np.asarray(waveforms)
    if wf.ndim != 2:
        raise ValueError("Waveforms must be a 2D array.")

    if samples_first is None and n_spikes_expected is not None:
        if wf.shape[1] == n_spikes_expected:
            samples_first = True
        elif wf.shape[0] == n_spikes_expected:
            samples_first = False

    if samples_first is None:
        if wf.shape[0] <= max_samples < wf.shape[1]:
            samples_first = True
        elif wf.shape[1] <= max_samples < wf.shape[0]:
            samples_first = False
        else:
            samples_first = wf.shape[0] <= wf.shape[1]

    transposed = False
    if not samples_first:
        wf = wf.T
        transposed = True

    n_samples, n_spikes = wf.shape
    if n_samples < min_samples or n_samples > max_samples:
        raise ValueError(f"Unexpected waveform length: {n_samples} samples.")
    if n_spikes < 1:
        raise ValueError("Waveform matrix has zero spikes.")

    wf = wf.astype(float, copy=False)

    finite_mask = np.isfinite(wf).all(axis=0)
    dropped = int((~finite_mask).sum())
    wf = wf[:, finite_mask]
    if wf.shape[1] == 0:
        raise ValueError("All waveforms contained NaN/Inf values.")

    info = {
        "transposed": transposed,
        "n_samples": wf.shape[0],
        "n_spikes": wf.shape[1],
        "dropped_spikes": dropped,
    }
    return wf, info


def align_spike_times(
    spike_times_s: np.ndarray,
    waveforms: np.ndarray,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Validate, sort, and filter spike times while keeping waveform alignment."""
    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    if times.size != waveforms.shape[1]:
        raise ValueError("Spike times length does not match waveform count.")

    finite_mask = np.isfinite(times)
    dropped_nan = int((~finite_mask).sum())
    times = times[finite_mask]
    wf = waveforms[:, finite_mask]

    out_of_range = 0
    if start_s is not None or stop_s is not None:
        mask = np.ones_like(times, dtype=bool)
        if start_s is not None:
            mask &= times >= start_s
        if stop_s is not None:
            mask &= times <= stop_s
        out_of_range = int((~mask).sum())
        times = times[mask]
        wf = wf[:, mask]

    order = np.argsort(times)
    sorted_needed = not np.all(order == np.arange(len(times)))
    if sorted_needed:
        times = times[order]
        wf = wf[:, order]

    info = {
        "dropped_nan": dropped_nan,
        "out_of_range": out_of_range,
        "sorted": sorted_needed,
        "n_spikes": times.size,
    }
    return times, wf, info


def convert_waveforms_to_uv(
    waveforms: np.ndarray,
    *,
    volt_threshold: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """Convert waveform amplitudes to microvolts if values look like Volts."""
    wf = np.asarray(waveforms, dtype=float)
    max_abs_before = float(np.nanmax(np.abs(wf))) if wf.size else 0.0
    converted = max_abs_before < volt_threshold
    if converted:
        wf = wf * 1e6
    max_abs_after = float(np.nanmax(np.abs(wf))) if wf.size else 0.0

    info = {
        "converted": converted,
        "units_label": "uV",
        "max_abs_before": max_abs_before,
        "max_abs_after": max_abs_after,
    }
    return wf, info


def mask_by_intervals(
    spike_times_s: np.ndarray,
    intervals: Iterable[Tuple[float, float]],
) -> np.ndarray:
    """Build a boolean mask for spikes that fall inside any interval."""
    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    mask = np.zeros(times.shape, dtype=bool)
    for start_s, stop_s in intervals:
        mask |= (times >= start_s) & (times <= stop_s)
    return mask


def sample_waveforms(
    waveforms: np.ndarray,
    n: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample waveform columns uniformly without replacement."""
    n_spikes = waveforms.shape[1]
    if n_spikes <= n:
        idx = np.arange(n_spikes)
        return waveforms, idx
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    idx = rng.choice(n_spikes, size=n, replace=False)
    return waveforms[:, idx], idx


def _amplitude_label(units_label: str) -> str:
    if units_label.lower() == "uv":
        return LABELS["amplitude_uv"]
    return f"Voltage ({units_label})"


def plotSpikes(
    waveforms: np.ndarray,
    *,
    n: int = 80,
    seed: int | None = None,
    fs: float = FS_SPIKE,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylim: Tuple[float, float] | None = None,
    units_label: str = "uV",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot random spike waveforms with an overlaid mean waveform."""
    wf = np.asarray(waveforms, dtype=float)
    wf_sample, _ = sample_waveforms(wf, n=n, seed=seed)
    mean_waveform = np.mean(wf, axis=1)

    t_ms = np.arange(wf.shape[0]) / fs * 1000.0

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(
        t_ms,
        wf_sample,
        color=COLORS["trace"],
        linewidth=LINEWIDTH_THIN,
        alpha=ALPHA_LIGHT,
    )
    ax.plot(
        t_ms,
        mean_waveform,
        color=COLORS["mean"],
        linewidth=LINEWIDTH_THICK,
        label="Mean",
    )
    ax.set_xlabel(LABELS["time_ms"])
    ax.set_ylabel(_amplitude_label(units_label))
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax
