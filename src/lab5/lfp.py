"""LFP spectrum helpers (Part 4)."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from .constants import FS_LFP, WELCH_NFFT, WELCH_OVERLAP_FRAC, WELCH_WINDOW_LEN
from .figures import save_figure
from .plot_style import FIGSIZE_STD, LABELS

def LFP_spectrum(
    lfp_data: dict,
    Fs: float = FS_LFP,
    *,
    channels: Iterable[int] = (1, 13, 20),
    save_prefix: str | None = None,
    out_dir=None,
) -> List[plt.Figure]:
    """Compute and plot LFP spectrum and spectrograms for stand/walk pairs."""
    figures: List[plt.Figure] = []
    window = signal.get_window("hann", WELCH_WINDOW_LEN)
    overlap = int(len(window) * WELCH_OVERLAP_FRAC)

    # ================= NEW: global dB color scale across ALL channels/conditions =================
    global_vals = []
    for ch in channels:
        stand_key = f"ch{ch:02d}Stand"
        walk_key = f"ch{ch:02d}Walk"

        stand = np.asarray(lfp_data[stand_key], dtype=float).squeeze()
        walk = np.asarray(lfp_data[walk_key], dtype=float).squeeze()

        f, tS, specS = signal.spectrogram(
            stand, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap,
            nfft=WELCH_NFFT, scaling="density", mode="psd"
        )
        _, tW, specW = signal.spectrogram(
            walk, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap,
            nfft=WELCH_NFFT, scaling="density", mode="psd"
        )

        specS_db = 10 * np.log10(specS + 1e-20)
        specW_db = 10 * np.log10(specW + 1e-20)
        global_vals.append(specS_db.ravel())
        global_vals.append(specW_db.ravel())

    global_vals = np.concatenate(global_vals)
    GLOBAL_VMIN = np.percentile(global_vals, 18)
    GLOBAL_VMAX = np.percentile(global_vals, 99)
    # ============================================================================================

    for ch in channels:
        stand_key = f"ch{ch:02d}Stand"
        walk_key = f"ch{ch:02d}Walk"

        stand = np.asarray(lfp_data[stand_key], dtype=float).squeeze()
        walk = np.asarray(lfp_data[walk_key], dtype=float).squeeze()

        # ================= Spectrum (Welch) =================
        fS, pS = signal.welch(
            stand, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap, nfft=WELCH_NFFT
        )
        fW, pW = signal.welch(
            walk, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap, nfft=WELCH_NFFT
        )

        fig1, ax1 = plt.subplots(figsize=FIGSIZE_STD)
        ax1.plot(fS, 10 * np.log10(pS + 1e-20), label="Stand")
        ax1.plot(fW, 10 * np.log10(pW + 1e-20), label="Walk")
        ax1.set_xlabel(LABELS["frequency_hz"])
        ax1.set_ylabel(LABELS["power_db"])
        ax1.set_title(f"ch{ch:02d} | LFP Spectrum (Welch)")
        ax1.set_xlim(0, 200)
        ax1.legend()
        fig1.tight_layout()
        figures.append(fig1)
        if save_prefix:
            save_figure(fig1, f"{save_prefix}_ch{ch:02d}_spectrum", out_dir=out_dir)

        # ================= Spectrogram =================
        f, tS, specS = signal.spectrogram(
            stand, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap,
            nfft=WELCH_NFFT, scaling="density", mode="psd"
        )
        _, tW, specW = signal.spectrogram(
            walk, fs=Fs, window=window,
            nperseg=len(window), noverlap=overlap,
            nfft=WELCH_NFFT, scaling="density", mode="psd"
        )

        specS_db = 10 * np.log10(specS + 1e-20)
        specW_db = 10 * np.log10(specW + 1e-20)

        fig2, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        im0 = axes[0].imshow(
            specS_db, aspect="auto", origin="lower",
            extent=[tS.min(), tS.max(), f.min(), f.max()],
            vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX
        )
        axes[0].set_ylim(1, 30)
        axes[0].set_xlabel(LABELS["time_s"])
        axes[0].set_ylabel(LABELS["frequency_hz"])
        axes[0].set_title("Stand")
        fig2.colorbar(im0, ax=axes[0], label=LABELS["power_db"])

        im1 = axes[1].imshow(
            specW_db, aspect="auto", origin="lower",
            extent=[tW.min(), tW.max(), f.min(), f.max()],
            vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX
        )
        axes[1].set_ylim(1, 30)
        axes[1].set_xlabel(LABELS["time_s"])
        axes[1].set_title("Walk")
        fig2.colorbar(im1, ax=axes[1], label=LABELS["power_db"])

        fig2.suptitle(f"ch{ch:02d} | LFP Spectrogram")
        fig2.tight_layout()
        figures.append(fig2)
        if save_prefix:
            save_figure(fig2, f"{save_prefix}_ch{ch:02d}_spectrogram", out_dir=out_dir)

    return figures
