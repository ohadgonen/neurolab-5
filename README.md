# Lab 5 - Extracellular Recordings

## Overview
This repo contains analysis code and a notebook report for Lab 5 extracellular recordings.

- `lab5.ipynb` is the main runner used to generate figures and summary statistics.
- `src/lab5/` contains the reusable analysis functions (no notebook-only logic).
- `plots/final/` contains the final figures used in the report.

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
1. Create/use a Python environment with `numpy`, `scipy`, and `matplotlib` (a repo-local venv works well for VS Code):
   - `python3 -m venv .venv && .venv/bin/python -m pip install -U pip`
   - `.venv/bin/python -m pip install numpy scipy matplotlib ipykernel`
2. Open `lab5.ipynb` in VS Code and select the `.venv` interpreter/kernel.
3. Run the notebook cells to regenerate figures into `plots/final/`.

If Matplotlib cache warnings appear, run with a writable config dir:
- `MPLCONFIGDIR=.mplconfig MPLBACKEND=Agg ...`

## Parts 2–3 report logic (what/why)
### Part 2 — Spike waveforms (same neuron?)
The protocol asks to compare spike waveforms between conditions and between different sorters (“same neuron?”). A practical issue is that unit labels (`a/b/c`) are not guaranteed to match between sorters, even when the same neuron is present.

We therefore generate:
- **Condition comparison (same recording context)** using the *dominant* unit (most spikes) in each condition file:
  - `plots/final/part2_ch20_stand_walk_compare_n80_fs40000_uV.png`
  - Rationale: within a file, “dominant unit” is a stable, single-unit representative without pooling different units.
- **People/sorter comparison** using a *matched pair* chosen by maximum correlation of the **mean waveform** across the available units:
  - `plots/final/part2_ch01_ohad_sagi_bestmatch_n80_fs40000_uV.png`
  - Rationale: avoids incorrect “same label ⇒ same neuron” assumptions and directly targets waveform similarity required by the protocol.

Implementation notes:
- Waveforms are normalized to `(n_samples, n_spikes)` (`ensure_waveforms_2d`), aligned with timestamps (`align_spike_times`), and converted to µV when needed (`convert_waveforms_to_uv`).
- Sampling of 80 traces is random by design (non-deterministic); the mean waveform is computed from all spikes.

### Part 3 — Firing rates (binning + statistics)
We compute firing rate in **non-overlapping bins** and compare per-bin rate samples statistically (Mann–Whitney by default), as requested in the protocol.

Figures used in the report:
- **Bin-size demo** (visualization uses non-overlapping bars to avoid overlap artifacts):
  - `plots/final/part3_bin_demo_SPK20a_bins_0p05_0p2_1p0_5p0_fs40000.png`
- **Stand vs Walk (SPK20a)** combined trace+hist:
  - `plots/final/part3_SPK20a_stand_walk_combo_bin0p2_fs40000.png`
- **Sorter comparison** uses a matched pair (by waveform similarity) for a fair “same neuron” firing-rate comparison:
  - `plots/final/part3_ohad_SPK01a_sagi_SPK01b_combo_bin0p2_fs40000.png`
- Summary stats:
  - `plots/final/part3_stats.csv` (includes `unit_key_a`/`unit_key_b`, `mean_a`/`mean_b`, `pvalue`, etc.)

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
- `sample_waveforms(waveforms, n, seed=None)`: sample waveform columns without replacement.
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
