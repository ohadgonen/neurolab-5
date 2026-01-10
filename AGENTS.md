# Agent context for Lab 5 (extracellular recordings)

## Purpose and current status
All required functions for Parts 1-4 exist in `src/lab5/`. Part 2-3 answers are summarized in `README.md`, and the remaining work is to rerun the notebook end-to-end in a local environment to confirm all outputs.

Keep `lab5.ipynb` as the runner/report, but put all reusable logic in closed Python functions under `src/lab5/` (no script-like top-level execution in modules).

## Project structure
- `lab5.ipynb`: report + figure generation (imports functions from `src/lab5/`).
- `src/lab5/`: analysis package (functions only).
  - `signal.py`: Part 1 (`plotSignal`).
  - `spikes.py`: Part 2 (`plotSpikes`) + waveform helpers.
  - `rates.py`: Part 3 (`firingRate`) + rate/stat helpers.
  - `lfp.py`: Part 4 (`LFP_spectrum`).
  - `io_mat.py`: `load_mat` helper.
  - `figures.py`: `apply_style()`, `save_figure()`, axis helpers.
  - `plot_style.py`: rcParams, sizes, colors, labels.
  - `constants.py`: sampling rates, filters, bin sizes, seeds.
  - `paths.py`: repo-relative `DATA_DIR`, `PLOTS_FINAL_DIR`.
- `matlab results/`: input `.mat` files.
- `plots/final/`: saved figures (via `save_figure()`).

## Data structure
- `matlab results/` contains stand/walk recordings for channels 01, 13, 20 from multiple people.
- Raw continuous signal: `SPK<ch>` with shape `(n_samples, 2)`.
- Spike times per unit: `SPK<ch><unit>` (e.g., `SPK01a`) with shape `(n_spikes, 1)`.
- Waveforms per unit: `SPK<ch><unit>_wf` with shape `(n_samples_per_waveform, n_spikes)` or transposed.
- Waveform timestamps: `SPK<ch><unit>_wf_ts`.
- Metadata: `SPK<ch>_ts`, `SPK<ch>_ts_step`, `SPK<ch>_ind`, `Start`, `Stop`.
- LFP: `LFP_data.mat` with `ch01Stand`, `ch01Walk`, `ch13Stand`, `ch13Walk`, `ch20Stand`, `ch20Walk`.
- Sampling rates: spike/raw 40,000 Hz and LFP 1,000 Hz (`src/lab5/constants.py`).

## Protocol questions to answer (Parts 2-3)
Part 2 (waveforms):
- Are individual waveforms similar or variable?
- Does the mean waveform represent the population well?
- Compare waveforms across stand vs walk on the same channel.
- Compare waveforms from different people on the same channel (same condition).

Part 3 (firing rates):
- How does bin size affect the firing-rate estimate? Show good and bad choices.
- Does firing rate change over the experiment?
- Rest vs movement differences (mean rates per stage).
- Statistical comparison using binned samples (histogram + p-value).
- Compare people on the same channel with the same bin size.

## Verification checklist
Terminal checks (automatable):
- Confirm required function definitions exist: `plotSignal`, `plotSpikes`, `firingRate`, `LFP_spectrum`.
- Load `.mat` files via `load_mat` and confirm key names/shapes.
- Confirm waveform orientation and units conversion when plotting.

Manual checks (requires human review):
- Run `lab5.ipynb` end-to-end to regenerate figures and confirm saved outputs in `plots/final/`.

## Plotting rules
- Use `apply_style()` once per notebook run.
- Save all figures with `save_figure()` into `plots/final/`.
- Keep modules function-only (no top-level execution).
