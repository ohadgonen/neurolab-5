"""Project-wide constants for Lab 5 analysis."""

FS_SPIKE = 40_000  # Hz
FS_LFP = 1_000  # Hz

# Part 1 filter defaults
HP_CUTOFF_HZ = 300.0
LP_CUTOFF_HZ = 300.0
FILTER_ORDER = 4

# Part 4 Welch defaults
WELCH_NFFT = 8192
WELCH_WINDOW_LEN = 600
WELCH_OVERLAP_FRAC = 0.25

# Part 3 firing-rate defaults (seconds)
BIN_SIZES_S = (0.05, 0.2, 1.0, 5.0)
PREFERRED_BIN_S = 0.2

DEFAULT_SEED = 0
