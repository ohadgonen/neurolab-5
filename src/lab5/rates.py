"""Firing-rate helpers (Part 3)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .constants import FS_SPIKE
from .plot_style import COLORS, FIGSIZE_STD, LABELS

__all__ = [
    "extract_units",
    "extract_stages",
    "infer_spike_times_seconds",
    "validate_spike_times",
    "nonoverlap_edges",
    "binned_firing_rate",
    "spike_train_counts",
    "plot_firing_rate",
    "firingRate",
    "stage_rate_samples",
    "compare_rate_samples",
]


def _is_spike_time_key(key: str) -> bool:
    if not key.startswith("SPK"):
        return False
    if key.endswith(("_wf", "_wf_ts", "_ts", "_ts_step", "_ind")):
        return False
    if not key[-1].isalpha():
        return False
    return True


def extract_units(mat: dict) -> Dict[str, np.ndarray]:
    """Extract spike-time arrays per unit from a loaded .mat dictionary."""
    units = {}
    for key, value in mat.items():
        if key.startswith("__") or not _is_spike_time_key(key):
            continue
        arr = np.asarray(value).squeeze()
        if arr.ndim != 1 or arr.size == 0:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        units[key] = arr.astype(float)
    return units


def _coerce_intervals(values: np.ndarray) -> List[Tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1 and arr.size == 2:
        return [(float(arr[0]), float(arr[1]))]
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return [(float(s), float(e)) for s, e in arr]
        if arr.shape[0] == 2 and arr.shape[1] <= 10:
            arr = arr.T
            return [(float(s), float(e)) for s, e in arr]
    return []


def extract_stages(mat: dict) -> Dict[str, List[Tuple[float, float]]]:
    """Extract stage intervals from a loaded .mat dictionary."""
    stages: Dict[str, List[Tuple[float, float]]] = {}
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        lower = key.lower()
        if "stand" in lower or "walk" in lower:
            intervals = _coerce_intervals(np.asarray(value))
            if intervals:
                name = "stand" if "stand" in lower else "walk"
                stages[name] = intervals

    if stages:
        return stages

    start = mat.get("Start") if "Start" in mat else mat.get("start")
    stop = mat.get("Stop") if "Stop" in mat else mat.get("stop")
    if start is not None and stop is not None:
        start_val = float(np.asarray(start).squeeze())
        stop_val = float(np.asarray(stop).squeeze())
        stages["recording"] = [(start_val, stop_val)]
    return stages


def infer_spike_times_seconds(
    spike_times: np.ndarray,
    *,
    Fs: float = FS_SPIKE,
    duration_s: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """Infer whether spike times are in samples or seconds and return seconds."""
    times = np.asarray(spike_times, dtype=float).reshape(-1)
    if times.size == 0:
        return times, {"units": "seconds", "converted": False, "max_time": 0.0}

    max_time = float(np.nanmax(times))
    looks_like_samples = False
    if duration_s is not None:
        looks_like_samples = max_time > duration_s * 2.0 and max_time > Fs
    else:
        looks_like_samples = max_time > Fs * 2.0

    if looks_like_samples:
        times = times / Fs

    info = {
        "units": "samples" if looks_like_samples else "seconds",
        "converted": looks_like_samples,
        "max_time": max_time,
    }
    return times, info


def validate_spike_times(
    spike_times_s: np.ndarray,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> np.ndarray:
    """Validate spike times (seconds), drop NaNs, sort, and apply bounds."""
    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    if times.size == 0:
        return times

    times = times[np.isfinite(times)]
    if start_s is not None:
        times = times[times >= start_s]
    if stop_s is not None:
        times = times[times < stop_s]

    if times.size > 1 and np.any(np.diff(times) < 0):
        times = np.sort(times)
    return times


def nonoverlap_edges(
    start_s: float,
    stop_s: float,
    bin_s: float,
) -> Tuple[np.ndarray, float]:
    """Compute non-overlapping bin edges and the effective stop time."""
    if bin_s <= 0:
        raise ValueError("bin_s must be positive.")
    duration = stop_s - start_s
    if duration <= 0:
        return np.array([]), start_s

    n_bins = int(np.floor(duration / bin_s))
    effective_stop_s = start_s + n_bins * bin_s
    if n_bins == 0:
        return np.array([]), effective_stop_s
    edges = np.linspace(start_s, effective_stop_s, n_bins + 1)
    return edges, effective_stop_s


def binned_firing_rate(
    spike_times_s: np.ndarray,
    *,
    start_s: float,
    stop_s: float,
    bin_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute firing rate in non-overlapping bins."""
    edges, effective_stop_s = nonoverlap_edges(start_s, stop_s, bin_s)
    if edges.size < 2:
        return np.array([]), np.array([]), np.array([]), edges

    counts, _ = np.histogram(spike_times_s, bins=edges)
    rate_hz = counts / bin_s
    t_centers_s = 0.5 * (edges[:-1] + edges[1:])
    return t_centers_s, rate_hz, counts, edges


def spike_train_counts(
    spike_times_s: np.ndarray,
    *,
    start_s: float,
    stop_s: float,
    Fs: float,
    bin_s: float,
) -> np.ndarray:
    """Compute per-bin counts using a sample-grid spike train."""
    if bin_s <= 0 or Fs <= 0:
        raise ValueError("Fs and bin_s must be positive.")

    duration_s = stop_s - start_s
    if duration_s <= 0:
        return np.array([])

    n_samples = int(round(duration_s * Fs))
    bin_samples = int(round(bin_s * Fs))
    if n_samples == 0 or bin_samples == 0:
        return np.array([])

    n_bins = n_samples // bin_samples
    if n_bins == 0:
        return np.array([])

    n_effective = n_bins * bin_samples
    train = np.zeros(n_effective, dtype=np.uint8)

    times = np.asarray(spike_times_s, dtype=float).reshape(-1)
    idx = np.floor((times - start_s) * Fs).astype(int)
    idx = idx[(idx >= 0) & (idx < n_effective)]
    train[idx] = 1

    counts = train.reshape(n_bins, bin_samples).sum(axis=1)
    return counts


def plot_firing_rate(
    t_centers_s: np.ndarray,
    rate_hz: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylim: Tuple[float, float] | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot firing rate vs time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    else:
        fig = ax.figure

    ax.plot(t_centers_s, rate_hz, color=COLORS["trace"])
    ax.set_xlabel(LABELS["time_s"])
    ax.set_ylabel(LABELS["rate_hz"])
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def firingRate(
    spike_times: np.ndarray,
    Fs: float,
    duration_s: float,
    window_size_s: float,
    *,
    start_s: float = 0.0,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute firing rate in non-overlapping bins with unit normalization."""
    spike_times_s, _ = infer_spike_times_seconds(spike_times, Fs=Fs, duration_s=duration_s)
    stop_s = start_s + duration_s
    times = validate_spike_times(spike_times_s, start_s=start_s, stop_s=stop_s)
    t_centers_s, rate_hz, counts, _ = binned_firing_rate(
        times,
        start_s=start_s,
        stop_s=stop_s,
        bin_s=window_size_s,
    )
    if ax is not None or title is not None:
        plot_firing_rate(t_centers_s, rate_hz, ax=ax, title=title)
    return t_centers_s, rate_hz, counts


def stage_rate_samples(
    spike_times_s: np.ndarray,
    intervals: Iterable[Tuple[float, float]],
    *,
    bin_s: float,
) -> np.ndarray:
    """Concatenate per-bin firing rates across intervals."""
    rates: List[np.ndarray] = []
    for start_s, stop_s in intervals:
        times = validate_spike_times(spike_times_s, start_s=start_s, stop_s=stop_s)
        _, rate_hz, _, _ = binned_firing_rate(
            times,
            start_s=start_s,
            stop_s=stop_s,
            bin_s=bin_s,
        )
        if rate_hz.size:
            rates.append(rate_hz)
    if not rates:
        return np.array([])
    return np.concatenate(rates)


def compare_rate_samples(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    *,
    method: str = "mannwhitney",
) -> dict:
    """Compare two per-bin rate samples with a two-sample test."""
    a = np.asarray(sample_a, dtype=float).reshape(-1)
    b = np.asarray(sample_b, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    result = {
        "method": method,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(np.mean(a)) if a.size else np.nan,
        "mean_b": float(np.mean(b)) if b.size else np.nan,
        "mean_diff": float(np.mean(a) - np.mean(b)) if a.size and b.size else np.nan,
        "pvalue": np.nan,
    }

    if a.size == 0 or b.size == 0:
        return result

    method = method.lower()
    if method in {"ttest", "ttest_ind"}:
        stat = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        result["pvalue"] = float(stat.pvalue)
        result["method"] = "ttest_ind"
    else:
        stat = stats.mannwhitneyu(a, b, alternative="two-sided")
        result["pvalue"] = float(stat.pvalue)
        result["method"] = "mannwhitney_u"
    return result
