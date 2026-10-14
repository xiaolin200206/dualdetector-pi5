# bench: co-located dual-detector benchmark for a Raspberry Pi 5

Measures latency, overlap, temperature, fan and (optionally) PMIC power of two ONNX detectors under
different release schedules. Derived from `deployment/detection.py` of Edge-Disease-Inference-Engine:
same Ultralytics + ONNX Runtime loading, camera acquisition, 82.0 degC software cut-off with a 5 s retry,
and masked decoding of `vcgencmd get_throttled`.

| File | Purpose |
|---|---|
| `dual_bench.py` | One run: one policy, one model pair; writes one directory. |
| `run_matrix.py` | The whole campaign: block-randomised order, cool-downs, resumable via `manifest.json`. |
| `sweep_matrix.py` | Release-period (1.0, 1.5, 2.0 s) by pest-offset (0, P/4, P/2) sweep, two blocks, 30 min runs. Imports `run_matrix`. |
| `run_night.sh` | Overnight queue: campaign3 (`serialp*`, 3 reps, 30 min) then sweep2 (dense offsets at P = 1.5 s, `--dense`, 2 reps, 30 min). Resumable. |
| `replay_matrix.py` | Replay test: still images shown one at a time in place of the camera (`--replay FOLDER --dwell 2.5 --gap 1.0`), five schedules, 2 reps, 20 min. Writes `replay.csv` per run; analysed by `analysis/make_replay.py`. |
| `perf_pairs.py` | `perf stat` helper for cache and bus counters. Tested with a stub only; never run on the Pi. |

## Conditions

| Name | Meaning |
|---|---|
| `indep` | Two loops, each sleeps after its own inference (0.8 s leaf, 1.2 s pest). `--start-offset 0.4` gives a staggered start. |
| `serial` | One loop: leaf, pest, sleep 1.2 s. |
| `sync` | Both detectors released together every `--period` s (1.5 by default). |
| `phase` | Same period; pest released `--phase-frac` of a period later (0.5 by default): no overlap, same work as `sync`. |
| `serialp` | Fixed period: leaf released every `--period` s, pest started the moment leaf returns. Same work rate as `sync`/`phase`; the fair sequential baseline (added for campaign3). |
| `sync_t2`, `phase_t2` | As above with 2 ONNX Runtime intra-op threads per model. |
| `sync_ss`, `phase_ss` | As above with the pest model replaced by `--pest-s-model` (YOLO11s). |

## On the Pi

```bash
sudo apt install v4l-utils
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # after every boot
python3 dual_bench.py --check            # sensors, fan, PMIC, camera
python3 dual_bench.py --policy phase --duration-min 5 --idle-s 10 --pmic \
    --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb --outdir trial
python3 run_matrix.py --dry-run --pest-s-model pest_s.onnx
tmux new -s exp
python3 run_matrix.py --outdir campaign1 --pest-s-model pest_s.onnx --duration-min 60 -- \
    --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb
```

Re-running the same command resumes an interrupted campaign: completed runs are skipped, the
interrupted run's directory is renamed `<label>__failed<n>` and the run is repeated.
Always pass `--duration-min` explicitly when resuming, so that the run length does not change.

## Conditions to keep constant

Same room, official 27 W supply (in the paper, 69 of the 77 runs were powered through a battery pack with pass-through charging; see the top-level README), Active Cooler on its default curve, no case, governor `performance`
(recorded in `meta.json`), camera fixed in place, nothing else running on the board. Room temperature is
not measured by default; a DS18B20 on 1-Wire is recorded automatically if present, and is worth adding.

## Power sampling

Use `--pmic-jitter 0.5` with fixed-period policies (the value used for the second round and the sweep). Without it the 2.0 s sampler can lock to the 1.5 s
release grid (see the top-level README).

Second round and sweep commands (as run):

```bash
python3 run_matrix.py --outdir campaign2 --conditions sync,phase,sync_t2,phase_t2,sync_ss,phase_ss \
    --reps 1 --seed 20261011 --pest-s-model pest_s.onnx --duration-min 60 --cool-min 10 --cool-max 25 -- \
    --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb --pmic-jitter 0.5
python3 sweep_matrix.py --outdir sweep1 -- \
    --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb --pmic-jitter 0.5
```

`--phase-s` gives the pest offset in seconds and overrides `--phase-frac`; `sweep_matrix.py --dense` uses it for
offsets 0.39, 0.42, 0.45, 0.50 and 0.60 s at P = 1.5 s (labels `p15_s390` etc., offset in ms).

## Replay mode

`dual_bench.py --replay FOLDER` replaces the camera with a `ReplaySource`: images are shown in a fixed
shuffled order, each for `--dwell` s followed by `--gap` s of a grey frame, repeating until the run ends.
`frame_id` in `inferences.csv` is then the presentation index and `frame_age_ms` the time since the image
appeared; `dets` lists the detected classes and confidences. `replay.csv` records every presentation.

```bash
python3 replay_matrix.py --outdir replay1 --duration-min 20 -- \
    --replay sabah --dwell 2.5 --gap 1.0 --leaf-model leaf_s.onnx --pest-model pest_n.onnx --pmic-jitter 0.5
```

`sabah/` is a folder of images (any layout; `.jpg/.png`), resized to 640 x 640 on load. The labels are not
needed on the Pi; `analysis/make_replay.py` reads them afterwards. Set the governor before resuming after a
reboot (`sudo cpufreq-set -g performance` or the sysfs write); `meta.json` records what was in force.

## Files written per run

`inferences.csv` (start, end, latency, frame id, scheduled time, lateness), `system.csv` (0.5 s: die
temperature, ARM clock, live and latched throttle bits, fan rpm and PWM, CPU, RAM, capture rate, threads),
`power.csv` (PMIC rails), `events.csv`, `meta.json` (arguments, versions, model checksums, thread
verification, camera settings, status).
