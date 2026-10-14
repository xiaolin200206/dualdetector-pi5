#!/bin/bash
# Overnight queue, added 12 October 2026:
#   campaign3: fixed-period serial baseline (serialp, serialp_t2, serialp_ss), 3 reps, 30 min
#   sweep2:    dense offset sweep at P = 1.5 s (0.39, 0.42, 0.45, 0.50, 0.60 s), 2 reps, 30 min
# Run inside tmux from ~/dualbench:   bash run_night.sh 2>&1 | tee night.log
# Re-running the same script resumes both (completed runs are skipped).
set -u
cd "$(dirname "$0")"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
MODELS="--leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb --pmic-jitter 0.5"
python3 run_matrix.py --outdir campaign3 --conditions serialp,serialp_t2,serialp_ss \
    --reps 3 --seed 20261013 --pest-s-model pest_s.onnx --duration-min 30 \
    --cool-min 5 --cool-max 12 -- $MODELS
python3 sweep_matrix.py --dense --outdir sweep2 --reps 2 --seed 20261014 --duration-min 30 -- $MODELS
echo "night queue finished: $(date)"
