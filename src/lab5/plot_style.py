"""Shared matplotlib styling settings for Lab 5 figures."""

FIGSIZE_WIDE = (10, 6)
FIGSIZE_STD = (7, 4)
DPI = 120

LINEWIDTH_THIN = 0.8
LINEWIDTH_MED = 1.4
LINEWIDTH_THICK = 2.2
ALPHA_LIGHT = 0.25
ALPHA_MED = 0.7

COLORS = {
    "raw": "#1f1f1f",
    "lowpass": "#1f77b4",
    "highpass": "#d62728",
    "mean": "#000000",
    "trace": "#1f77b4",
}

LABELS = {
    "time_s": "Time (s)",
    "time_ms": "Time (ms)",
    "amplitude_uv": "Voltage (uV)",
    "rate_hz": "Firing rate (spikes/s)",
    "frequency_hz": "Frequency (Hz)",
    "power_db": "Power (dB)",
}

RC_PARAMS = {
    "figure.dpi": DPI,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": LINEWIDTH_MED,
}
