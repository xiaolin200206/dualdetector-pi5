# Two detectors on one Raspberry Pi 5: release timing, latency, temperature and power

Code, analysis scripts and raw logs for the paper

> Lin, D. S. (2026). Release timing of two YOLO11 detectors sharing a Raspberry Pi 5 on a crop-scouting
> unit. Manuscript under review; the full citation will be added on acceptance.

A leaf-disease detector (YOLO11s) and a pest detector (YOLO11n or YOLO11s), trained on durian images,
run in ONNX Runtime on the CPU of one Raspberry Pi 5 with the Active Cooler. Every inference is
timestamped at start and end, so overlap between the detectors is observed, not inferred. 77 runs in
six sets:

| Set | Runs | Run length, steady window | What it tests |
|---|---|---|---|
| Main campaign (`campaign1`) | 8 policies × 3 = 24 | 60 min, min 20–60 | penalty of overlap at equal work; thermal effect |
| Second round (`campaign2`) | 6 fixed-period policies × 1 | 60 min, min 20–60 | repeat in a separate session; power with randomised sampling |
| First sweep (`sweep1`) | P = 1.0, 1.5, 2.0 s × offset 0, P/4, P/2 × 2 = 18 | 30 min, min 10–30 | non-overlap condition, Eq. (1) |
| Fixed-period serial, `serialp` (`campaign3`) | leaf every 1.5 s, pest when leaf returns; 3 configurations × 3 = 9 | 30 min, min 10–30 | same work rate as sync/phase, pest chained after leaf |
| Second sweep (`sweep2`) | P = 1.5 s, offset 0.39–0.60 s × 2 = 10 | 30 min, min 10–30 | where contact stops |
| Replay (`replay1`) | 5 schedules × 2 = 10 | 20 min, min 2–20 | detection outcomes and time from image onset to answers, with 281 labelled images from two other orchards; Eq. (3) |

## Main results (from `results/tables`)

| Policy (main campaign unless stated) | Leaf median (IQR), ms | Pest median (IQR), ms | Both answers, ms | Leaf inferences overlapped |
|---|---|---|---|---|
| Independent timers (two loops) | 393 (152) | 161 (208) | – | 37 % |
| Serialised cycle (lower work rate) | 387 (11) | 157 (6) | 544 (from leaf start) | 0 |
| Synchronous release, P = 1.5 s | 715 (127) | 554 (281) | 716 (from release) | 100 % |
| Phased release, pest 0.75 s later | 388 (11) | 154 (6) | 904 (from release, includes the offset) | 0 |
| Fixed-period serial (serial baseline set) | 386 | 157 | 544 (from release) | 0 |
| Phased, offset 0.42 s (second sweep) | 387 | 159 | 580 (from release) | 0.1 % (one inference) |

Synchronous and phased release have the same models and work rate (40 inferences per minute per
detector). No run reached the 82 °C software cut-off or set a throttle flag; the hottest sample was 74.9 °C.

- **Independent timers.** Overlapped leaf inferences took 58 % longer, in bursts at the beat period of the
  two loops (predicted 9.68–9.82 s, spectral peak 9.75–9.83 s). The overlap estimate
  `(L_leaf + L_pest) / P_other` gives 37.3 % (leaf) and 42.8–42.9 % (pest) against 36.9–37.8 % and
  42.5–43.4 % observed (`results/tables/overlap_estimate.csv`).
- **Non-overlap condition.** `offset >= L_leaf and offset + L_pest <= period`. With overlap-free medians
  (0.388, 0.154 s) it agrees with all nine settings of the first sweep but fails at 0.39 s in the second
  sweep (36 % of leaf inferences still overlapped). With overlap-free p99 latencies (0.410, 0.166 s) it agrees
  with all 14 settings of both sweeps (at most one overlapped inference per run where it predicts none), and
  the overlapped share at shorter offsets follows P(L_leaf > offset) (`results/tables/offset_rule.csv`).
- **Replay timing.** The measured wait from image onset to the next leaf release is close to uniform on
  [0, 1.5) s (median 0.72–0.76 s; `results/tables/wait_distribution.csv`), as Eq. (3) assumes.
- **Two threads per model** removed most of the jitter, not the slowdown; it is the one configuration in
  which synchronous release delivered both answers sooner (612 ms) than fixed-period serial (698 ms).
- **Power** (second round, randomised sampling, single runs): overlap raised board rail power by 3.2 to 12.8 %.
- **Replay.** The detection outcome did not depend on the schedule (identical on every shared
  presentation within a replicate). Median time from image onset to both answers: 1.20–1.23 s
  (independent timers, higher work rate), 1.24 s (fixed-period serial), 1.32 s (phase 0.42 s),
  1.36–1.38 s (phase 0.75 s), 1.41–1.42 s (synchronous). The wait-plus-service relation predicts
  the medians of the eight fixed-period runs within 0.06 s from camera-run latencies, and within 0.024 s from
  the latencies of the same runs (`results/tables/onset_model.csv`; dwell coverage in `dwell_coverage.csv`).
  The overlap estimate, applied out of sample to the two replay independent-timer runs, gives 35.1 % / 40.9 %
  against 34.2–35.2 % / 39.8–41.0 % observed.
  Miss and false-alarm rates on the replay set (leaf 14–15 % / 29–30 %, pest 35 % / 59–60 % at conf 0.2)
  describe the models on out-of-region images and are not a result of the study.

The timing, thermal and power numbers of the paper are produced by the scripts in `analysis/` from the raw
logs; Table 1 (detector size and validation accuracy) comes from `data_rebuild/` and `models/table1.json`.

## Layout

| Folder | Contents |
|---|---|
| `bench/` | Runs on the Pi. `dual_bench.py` (one run; `--replay FOLDER` replaces the camera with still images), `run_matrix.py` (randomised campaign with cool-downs, resumable), `sweep_matrix.py` (period and offset sweep), `run_night.sh` (serial baseline and second sweep), `replay_matrix.py` (replay test), `perf_pairs.py` (hardware-counter helper; tested with a stub only, never run on the Pi, not used in the paper). |
| `analysis/` | Runs anywhere. `analyse.py` (per-run metrics), `make_results.py` (main campaign: Tables 4–5, Figs 2–6), `make_round2_sweep.py` (second round, first sweep: Tables 6 and S1, Fig. 7), `make_campaign3_sweep2.py` (serial baseline, second sweep: Table 7), `make_replay.py` (replay: Table 8), `make_models.py` (Eqs 1–3 against the logs, dwell coverage, Fig. 8), `make_fig_schematic.py` (Fig. 1). |
| `data/` | Raw logs, one zip per set (inference, 0.5 s system log, PMIC log, events, metadata, and `replay.csv` for the replay runs). Interrupted attempts are kept as `<label>__failed<n>` and ignored by the scripts. `sabah_labels/`: YOLO label files, class list and image list of the 281 replay images (the images are archived at https://doi.org/10.5281/zenodo.22030623 and are not redistributed here). |
| `results/` | Tables (`tables/`) and figures (`figures/fig01`–`fig08`, PNG and PDF) generated from the raw logs. |
| `data_rebuild/` | Leak-audited dataset rebuild, training and ONNX export (workstation with a CUDA GPU). |
| `models/` | The three ONNX models used on the Pi (`leaf_s.onnx`, `pest_n.onnx`, `pest_s.onnx`), the training `manifest.json`, and `table1.json` (size, compute, validation accuracy and MD5 of each model; Table 1). |

## Reproduce the tables and figures (no Pi needed)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy==2.5.3 pandas==3.0.5 matplotlib==3.11.2 scipy==1.18.1
for z in campaign1 campaign2 sweep1 campaign3 sweep2 replay1; do unzip -q data/${z}_raw.zip -d data; done
python3 -I analysis/make_results.py data/campaign1 results --warmup-min 20 --max-min 60
python3 -I analysis/make_round2_sweep.py --round1 results/tables/runs.csv \
    --round2 data/campaign2 --sweep data/sweep1 --out results
python3 -I analysis/make_campaign3_sweep2.py --round1 results/tables/runs.csv \
    --sweep1 results/tables/sweep_runs.csv --campaign3 data/campaign3 --sweep2 data/sweep2 --out results
python3 -I analysis/make_replay.py --runs data/replay1 --images data/sabah_labels --out results
python3 -I analysis/make_models.py --tables results/tables --campaign1 data/campaign1 \
    --replay data/replay1 --out results
python3 -I analysis/make_fig_schematic.py --out results
```

Run the scripts in this order; later scripts read tables written by earlier ones. The tables are
reproduced byte for byte from the archived logs (checked with Python 3.13, numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, scipy 1.18.1); the figure PDFs carry no
creation date, so they reproduce too.
`make_replay.py` needs only the label files; pass the image folder instead of `data/sabah_labels` if you
have it (same result).

`--max-min 60` analyses the first 60 min of every run. Three runs of the main campaign
(`phase_t2_r3`, `serial_r3`, `sync_r3`) were recorded for 90 min because the campaign was resumed after
a power loss with a different default length; their first 60 min follow the same protocol.

## Re-run the benchmark on a Pi 5

See `bench/README.md`. The command used for the main campaign was

```bash
python3 bench/run_matrix.py --outdir campaign1 --pest-s-model pest_s.onnx --duration-min 60 -- \
  --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb
```

## Power sampling

The PMIC was read every 2.0 s on a fixed grid in the main campaign, and the fixed-period policies
release on a 1.5 s grid. The two grids lock together, so the readings fall at the same few positions of
every period and the mean is biased (for example, none of the readings in the phased runs fell inside a
pest inference although those inferences take 10 % of the time). `analyse.py` compares, for each
detector, the share of readings inside an inference with the share of time inside one, and sets power to
NaN when they differ by more than 0.03. In the main campaign only the independent-timer and serialised
runs pass. All later sets used `--pmic-jitter 0.5` (a random 0–0.5 s added to each interval): all 6
second-round runs and 16 of the 18 first-sweep runs pass; in the serial baseline and the second sweep only
3 of 9 and 2 of 10 pass, and power is not reported for those sets. Power is the sum of the PMIC rail
readings, not wall or battery power.

## Power supply and interruptions

The board was powered through a battery pack with pass-through charging (the same arrangement as in Lin (2026), *Smart Agricultural Technology* 15, 102526, https://doi.org/10.1016/j.atech.2026.102526)
for 69 of the 77 completed runs: charger connected for the first 51, battery alone for the 12 runs from 00:59
to 08:04 on 13 October, and supply path not recorded for four serial-baseline runs (21:45 on 12 October to
00:04 on 13 October) and the first two replay runs. Nine attempts ended abruptly, without the
`RUN_END` event that every completed run writes and without any error event, which is consistent with a
loss of power. Seven left logs that end after a complete line followed by zero-filled blocks, one
(`serialp_t2_r3__failed2`) left empty files and one (replay) left none. In `rp_sync_r1__failed2` the throttle flags show under-voltage and the clock capped at 1000 MHz in
the minute before the cut; for the others the cause was not recorded:

| Set | Interrupted attempts | Kept as |
|---|---|---|
| Main campaign | 1 (10 Oct, about 11:50) | `phase_t2_r3__failed1` |
| Fixed-period serial | 5 (12 Oct 13:27, 20:54, 21:40; 13 Oct about 00:31 and before 00:59) | `serialp_r2__failed1–3`, `serialp_t2_r3__failed1–2` |
| Replay | 3 (13 Oct, about 15:54 and 16:08; one between 17:24 and 21:30 that wrote no files) | `rp_sync_r1__failed1–2` |

After the losses of 12–13 October the charger was disconnected, and the 12 runs that followed (to 08:04
on 13 October) ran on the battery alone. After the replay losses
the battery pack was removed, and the last eight replay runs took the official 27 W supply directly
(21:30 to 00:55) without interruption. The PMIC flagged no under-voltage in any completed run.

The reboot after the last power loss reset the CPU governor to `ondemand`, and it was not set back to
`performance` before those eight replay runs (recorded in each run's `meta.json`; the first two replay
runs and all 67 earlier runs record `performance`). The clock read below 2,400 MHz in 5 to 17 % of 0.5 s
samples of their steady windows, mostly between inferences or in their first milliseconds; per-call latencies match the performance runs of
the same schedule to within 4 % (leaf) and 7 % (pest) and the replay times to within 0.01 s (`results/tables/replay_runs.csv`).

## Not done

Frame staleness with a moving scene, shorter dwell times, hardware performance counters, other models,
boards or runtimes, INT8, and an outdoor or field test were not run. The replay test checks detection
outcomes at one dwell time (2.5 s) and one period (1.5 s) only.

## Dataset and models

- Training images (leaf-disease and pest exports, 1372 unique images): https://doi.org/10.5281/zenodo.22646875 (CC BY 4.0).
- Replay images (281 Sabah images, inside the farm- and tree-level dataset): https://doi.org/10.5281/zenodo.22030623 (CC BY-NC 4.0).
- The three trained ONNX models (`leaf_s.onnx`, `pest_n.onnx`, `pest_s.onnx`) and the training `manifest.json` are in `models/`; their MD5 checksums are listed in `models/table1.json` and recorded in every run's `meta.json`.
- `data_rebuild/` reproduces the split and the training from the training images.

## Licence

Code: MIT (`LICENSE`). Ultralytics (AGPL-3.0) is used as a library and is not included.
Raw logs (`data/*_raw.zip`), result tables (`results/`) and the models in `models/`: CC BY 4.0.
`data/sabah_labels/`: CC BY-NC 4.0, the licence of the dataset the labels come from (zenodo.22030623).
Images: see the dataset records above.
