#!/usr/bin/env python3
"""
03_train.py - train and export the three detectors used by dual_bench.py.

  leaf_s   YOLO11s on built/leaf
  pest_n   YOLO11n on built/pest
  pest_s   YOLO11s on built/pest     (backbone-matched pair, for sync_ss / phase_ss)

One frozen recipe for all three, the same one the TII study used, except the
batch size (6 GB laptop GPU) and a single seed: this study compares schedules,
not detectors, so accuracy is reported but not compared across seeds.

The DEPLOYED artefact is the final-epoch weight (last.pt), not the best-epoch
one: best.pt is selected on the validation split, which would then serve twice.

Outputs, in --models:
  leaf_s.onnx, pest_n.onnx, pest_s.onnx      opset 12, static 640x640, batch 1, no NMS
  manifest.json                              checksums, versions, recipe, val metrics

  python 03_train.py --data built
  python 03_train.py --data built --only pest_n      # one model
  python 03_train.py --data built --resume           # continue interrupted runs
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

PLAN = {
    "leaf_s": ("yolo11s.pt", "leaf"),
    "pest_n": ("yolo11n.pt", "pest"),
    "pest_s": ("yolo11s.pt", "pest"),
}

RECIPE = dict(epochs=150, imgsz=640, optimizer="AdamW", lr0=1e-3, lrf=0.01,
              momentum=0.9, weight_decay=5e-4, warmup_epochs=3, close_mosaic=10,
              patience=0,            # 0 disables early stopping: every run completes its budget
              deterministic=True, seed=0, amp=True, plots=True, exist_ok=False)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="built")
    ap.add_argument("--models", default="models")
    ap.add_argument("--project", default="runs_train")
    ap.add_argument("--only", default=None, help="comma list from: " + ",".join(PLAN))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    data = Path(a.data)
    audit = json.load(open(data / "audit.json", encoding="utf-8"))
    if any(audit["fails"].values()):
        sys.exit(f"{data}/audit.json reports leaks {audit['fails']}; refusing to train.")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available: check the PyTorch install (needs the CUDA build).")

    models = Path(a.models)
    models.mkdir(exist_ok=True)
    mpath = models / "manifest.json"
    manifest = json.load(open(mpath)) if mpath.exists() else {}
    manifest["environment"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__, "platform": platform.platform()}
    manifest["dataset_audit_md5"] = md5(data / "audit.json")
    manifest["recipe"] = {**RECIPE, "epochs": a.epochs, "batch": a.batch, "workers": a.workers}

    todo = a.only.split(",") if a.only else list(PLAN)
    for key in todo:
        base, dom = PLAN[key]
        yaml_path = (data / dom / "data.yaml").resolve()
        run_dir = Path(a.project) / key
        last = run_dir / "weights" / "last.pt"
        t0 = time.time()
        if a.resume and last.exists():
            print(f"[{key}] resuming {last}")
            YOLO(str(last)).train(resume=True)
        elif run_dir.exists():
            sys.exit(f"{run_dir} exists; use --resume, or remove it to retrain {key}.")
        else:
            print(f"[{key}] training {base} on {yaml_path}")
            YOLO(base).train(data=str(yaml_path), batch=a.batch, workers=a.workers,
                             device=a.device, project=str(Path(a.project).resolve()),
                             name=key, **{**RECIPE, "epochs": a.epochs})
        if not last.exists():
            sys.exit(f"[{key}] no {last}; training did not finish")

        m = YOLO(str(last))
        met = m.val(data=str(yaml_path), imgsz=640, batch=a.batch, split="val",
                    device=a.device, plots=False)
        onnx = Path(m.export(format="onnx", opset=12, dynamic=False, simplify=True,
                             imgsz=640, batch=1, nms=False, device="cpu"))
        dst = models / f"{key}.onnx"
        shutil.copy2(onnx, dst)
        try:
            per_class = {m.names[i]: float(v) for i, v in
                         zip(met.box.ap_class_index, met.box.ap50)}
        except Exception:
            per_class = None
        manifest[key] = {
            "base_weights": base, "dataset": str(yaml_path), "run_dir": str(run_dir.resolve()),
            "deployed_checkpoint": "last.pt (final epoch)",
            "last_pt_md5": md5(last), "onnx_md5": md5(dst),
            "val_framework_validator": {"map50": float(met.box.map50), "map50_95": float(met.box.map)},
            "per_class_ap50": per_class,
            "train_hours": round((time.time() - t0) / 3600, 2)}
        json.dump(manifest, open(mpath, "w"), indent=2)
        print(f"[{key}] mAP50 {met.box.map50:.3f}  ->  {dst}  md5 {manifest[key]['onnx_md5']}")

    print(f"done -> {mpath}")


if __name__ == "__main__":      # required on Windows (DataLoader workers spawn)
    main()
