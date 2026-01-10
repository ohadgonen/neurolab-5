# Lab 5 - Extracellular Recordings

## Overview
This repo contains analysis code and a notebook report for Lab 5 extracellular recordings. The notebook `lab5.ipynb` is the runner/report, and all reusable logic lives in `src/lab5/` as closed Python functions.

## Project layout
- `lab5.ipynb`: report + figure generation (imports from `src/lab5/`).
- `src/lab5/`: analysis package (functions only).
  - `signal.py`: Part 1 signal filtering and plotting.
  - `spikes.py`: Part 2 spike waveform helpers and plotting.
  - `rates.py`: Part 3 firing-rate helpers and stats.
  - `lfp.py`: Part 4 LFP spectrum and spectrograms.
  - `io_mat.py`: `.mat` loading helpers.
  - `figures.py`: shared plotting and saving helpers.
  - `plot_style.py`: figure sizes, colors, labels, rcParams.
  - `constants.py`: sampling rates, filter defaults, bin sizes.
  - `paths.py`: repo-relative data and output paths.
- `matlab results/`: input `.mat` data files.
- `plots/final/`: saved figures (created by `save_figure()`).

## Data layout
- `matlab results/` contains stand/walk recordings for channels 01, 13, 20 from multiple people.
- Raw continuous signal: `SPK<ch>` with shape `(n_samples, 2)`.
- Spike times per unit: `SPK<ch><unit>` (e.g., `SPK01a`), shape `(n_spikes, 1)`.
- Waveforms per unit: `SPK<ch><unit>_wf`, shape `(n_samples_per_waveform, n_spikes)` or transposed.
- Waveform timestamps: `SPK<ch><unit>_wf_ts`.
- Metadata: `SPK<ch>_ts`, `SPK<ch>_ts_step`, `SPK<ch>_ind`, `Start`, `Stop`.
- LFP: `LFP_data.mat` contains `ch01Stand`, `ch01Walk`, `ch13Stand`, `ch13Walk`, `ch20Stand`, `ch20Walk`.

Sampling rates and defaults live in `src/lab5/constants.py`:
- Spike/raw sampling rate: `FS_SPIKE = 40_000`.
- LFP sampling rate: `FS_LFP = 1_000`.

## How to run
1. Ensure a Python environment with `numpy`, `scipy`, and `matplotlib`.
2. Open `lab5.ipynb`.
3. In the first cell, import helpers and call `apply_style()`.
4. Select the desired `.mat` files and units, then run the notebook cells.
5. Use `save_figure()` to write outputs to `plots/final/`.

For protocol questions and manual verification steps, see `AGENTS.md`.

## Protocol answers (Parts 2-3)
Part 2 (waveforms; see `plots/final/part2_ch20_stand_walk_compare_n80_seed0_fs40000_uV.png` and `plots/final/part2_ch01_ohad_sagi_compare_n80_seed0_fs40000_uV.png`):
- Individual waveforms cluster tightly with moderate amplitude spread; the mean waveform captures the central shape well.
- Stand vs Walk on Ch20 (Carmel): waveform shape and timing are highly similar; Walk shows a slightly deeper negative trough with comparable repolarization.
- Ohad vs Sagi on Ch01 (Stand): overall shape is consistent across sorters; Sagi shows slightly larger peak-to-peak amplitude and broader variability.

Part 3 (firing rates; bin size = 0.2 s unless noted):
- Bin-size demo (`plots/final/part3_bin_demo_SPK20a_bins_0p05_0p2_1p0_5p0_fs40000.png`): 0.05 s is too spiky/noisy, 1-5 s oversmooths, 0.2 s balances resolution and stability.
- Stand vs Walk (Ch20 SPK20a; `plots/final/part3_SPK20a_stand_walk_bin0p2_fs40000.png`): mean stand = 4.816 spikes/s, mean walk = 5.843 spikes/s, mean diff (stand - walk) = -1.027 spikes/s, Mann-Whitney p = 6.03e-10 (stats in `plots/final/part3_stats.csv`). Histogram of walk - stand differences is right-shifted (`plots/final/part3_SPK20a_walk_minus_stand_hist_bin0p2_fs40000.png`).
- Ohad vs Sagi (Ch01 SPK01a, Stand; `plots/final/part3_SPK01a_ohad_sagi_bin0p2_fs40000.png`): mean Ohad = 2.913 spikes/s, mean Sagi = 4.015 spikes/s, mean diff (Ohad - Sagi) = -1.102 spikes/s, Mann-Whitney p = 2.08e-13. Histogram of Sagi - Ohad differences is right-shifted (`plots/final/part3_SPK01a_sagi_minus_ohad_hist_bin0p2_fs40000.png`).
- Within-condition traces show no obvious monotonic drift; Walk has more frequent higher-rate bins than Stand.

## Function reference (brief)
### `src/lab5/io_mat.py`
- `load_mat(path)`: load a `.mat` file with consistent options.

### `src/lab5/figures.py`
- `apply_style()`: apply shared matplotlib rcParams.
- `save_figure(fig, name, ...)`: save figures to `plots/final/`.
- `set_shared_limits(axes, ...)`: set shared x/y limits across axes.
- `compute_global_ylim(curves, padding=0.05)`: compute shared y-limits with padding.

### `src/lab5/signal.py`
- `plotSignal(raw_signal, ...)`: plot raw, low-pass, and high-pass signal segments.

### `src/lab5/spikes.py`
- `ensure_waveforms_2d(waveforms, ...)`: normalize waveform arrays to `(n_samples, n_spikes)`.
- `align_spike_times(spike_times_s, waveforms, ...)`: sort/filter spike times and keep waveform alignment.
- `convert_waveforms_to_uv(waveforms, ...)`: convert waveform amplitudes to microvolts when needed.
- `mask_by_intervals(spike_times_s, intervals)`: boolean mask for spikes inside time intervals.
- `sample_waveforms(waveforms, n, seed=0)`: sample waveform columns without replacement.
- `plotSpikes(waveforms, ...)`: plot sampled waveforms with an overlaid mean waveform.

### `src/lab5/rates.py`
- `extract_units(mat)`: extract spike-time arrays per unit from a `.mat` dict.
- `extract_stages(mat)`: extract stand/walk (or recording) intervals from a `.mat` dict.
- `infer_spike_times_seconds(spike_times, Fs, ...)`: infer samples vs seconds and return seconds.
- `validate_spike_times(spike_times_s, ...)`: clean and sort spike times; apply bounds.
- `nonoverlap_edges(start_s, stop_s, bin_s)`: compute non-overlapping bin edges.
- `binned_firing_rate(spike_times_s, ...)`: firing rate in non-overlapping bins.
- `spike_train_counts(spike_times_s, ...)`: per-bin counts via a sample-grid spike train.
- `plot_firing_rate(t_centers_s, rate_hz, ...)`: plot firing rate vs time.
- `firingRate(spike_times, Fs, duration_s, window_size_s, ...)`: end-to-end binned rate computation.
- `stage_rate_samples(spike_times_s, intervals, bin_s)`: concatenate per-bin rates for intervals.
- `compare_rate_samples(sample_a, sample_b, ...)`: two-sample test on binned rates.

### `src/lab5/lfp.py`
- `LFP_spectrum(lfp_data, ...)`: Welch spectra and spectrograms for stand/walk LFP.
