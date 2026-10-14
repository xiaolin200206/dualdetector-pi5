#!/usr/bin/env python3
"""
04_table1.py - collect the numbers for Table 1 (parameters, GFLOPs, ONNX size, validation metrics).

  python 04_table1.py --models result\\onnx --runs result\\runs_train

Reads models/manifest.json (written by 03_train.py) and the final-epoch checkpoints
runs_train/<name>/weights/last.pt, and prints one JSON object per detector.
Parameters and GFLOPs come from the Ultralytics model summary of the trained checkpoint.
Parameters are counted directly from the checkpoint's network and GFLOPs come from
ultralytics.utils.torch_utils.get_flops (one 640x640 image), because YOLO(...).info() returns
None in some Ultralytics versions.
"""
import argparse
import json
from pathlib import Path

NAMES = ["leaf_s", "pest_n", "pest_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--runs", default="runs_train")
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    man = json.load(open(Path(a.models) / "manifest.json", encoding="utf-8"))
    out = {}
    for n in NAMES:
        last = Path(a.runs) / n / "weights" / "last.pt"
        yolo = YOLO(str(last))
        net = yolo.model
        params = sum(p.numel() for p in net.parameters())
        gflops = get_flops(net, imgsz=640)   # GFLOPs for one 640x640 image
        onnx = Path(a.models) / f"{n}.onnx"
        met = man[n]["val_framework_validator"]
        out[n] = {
            "parameters_M": round(params / 1e6, 2),
            "gflops": round(gflops, 1),
            "onnx_mb": round(onnx.stat().st_size / 2**20, 1),
            "map50": round(met["map50"], 3),
            "map50_95": round(met["map50_95"], 3),
            "onnx_md5": man[n]["onnx_md5"],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
