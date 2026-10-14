# Dataset rebuild and model training

Rebuilds a leak-audited train/validation split from the two annotated YOLO exports (leaf and
pest), trains the three detectors used by the benchmark, and exports them to ONNX. Run on a
workstation with a CUDA GPU and a Python environment that has `ultralytics`.

```bat
python 01_inspect.py --leaf exports\leaf --pest exports\pest --background exports\background --out result\inspect_out
python 02_build.py   --inspect result\inspect_out --out result\built --class-map class_map.yaml
python 03_train.py   --data result\built --models result\onnx --project result\runs_train
python 04_table1.py  --models result\onnx --runs result\runs_train
```

1. **01_inspect.py** (read-only). Inventory of images and labels, SHA-256 and perceptual-hash
   (64-bit DCT, all eight rotations and flips) duplicates, capture bursts, video frames, label
   problems. Writes `report.md`, `pairs.csv`, `inventory.csv`, `hashes.npy`, `labels.json`.
2. **02_build.py**. Merges byte-identical files (box sets unioned), groups images (same export name,
   same video, consecutive camera counters with gap <= 3, perceptual hash <= 2), assigns whole
   groups to train or validation (20 %), rarest class first, projects the split onto a leaf and a
   pest set, and audits the boundary. The build refuses to finish if anything leaks.
   Writes `audit.md`/`audit.json`, `split.csv` and the two YOLO datasets.
3. **03_train.py**. One frozen recipe for all three models (150 epochs, AdamW, seed 0, batch 8, no
   early stopping), final-epoch weights deployed, ONNX export (opset 12, static 640, batch 1, no NMS).
   Writes `<name>.onnx` and `manifest.json` (checksums, versions, recipe, validation metrics).
4. **04_table1.py**. Collects parameters, GFLOPs, ONNX size and validation metrics for Table 1 of the paper.

5. **05_check_external.py**. Checks that an external image set (the Sabah replay images) is independent of
   the training and validation images: SHA-256 matches and the nearest perceptual hash under the eight
   rotations and flips.

`common.py` holds shared helpers. `class_map.yaml` is the class map used for the paper (defaults: no renames or drops).

The leak checks (export stem, SHA-256, video, perceptual hash within the grouping threshold, camera
burst) must all be zero for `03_train.py` to start.
