#!/usr/bin/env python3
"""
dual_bench.py - co-located dual-detector benchmark for a Raspberry Pi 5.

Derived from deployment/detection.py in Edge-Disease-Inference-Engine and kept
deliberately close to it: the same model loading (Ultralytics over an ONNX
export, executed by ONNX Runtime on the CPU), the same camera acquisition with
the same refusal to run without a camera, the same 82.0 C software cut-off with
a bare 5 s retry, and the same masked decoding of `vcgencmd get_throttled`.

What is new is what the earlier engine could not measure:

  * two detectors (leaf-disease and pest) under one of four scheduling policies;
  * the start AND end time of every inference on one monotonic clock, so that
    overlap between the two detectors is observed rather than inferred;
  * a 0.5 s system sampler: die temperature, ARM clock as the firmware reports
    it, live and sticky throttle bits, fan RPM and PWM, CPU and RAM use, the
    capture rate, and an optional DS18B20 room thermometer;
  * optional SoC power from `vcgencmd pmic_read_adc` every 2 s;
  * an idle baseline before inference starts (ambient proxy and power floor);
  * optional control of ONNX Runtime's intra-op thread count, to separate CPU
    oversubscription from memory-system contention.

Policies
--------
  indep   Legacy independent timers, as in the original scheduler: each worker
          runs one inference, then sleeps a fixed interval (0.8 s leaf, 1.2 s
          pest). Periods are latency + sleep, so they are incommensurate and the
          two windows coincide on a beat period. --start-offset 0.4 reproduces
          the old "staggered" policy; 0 reproduces the old "parallel" one.
  serial  Legacy serialised cycle: leaf, then pest, then sleep 1.2 s.
  sync    Fixed-rate: both detectors are released at the same instant every
          --period seconds. Maximal overlap at a known work rate.
  serialp Fixed-rate serial: every --period seconds the leaf detector is released
          and the pest detector starts the moment it returns. Same work rate as
          sync and phase; the fair "one after the other" baseline.
  phase   Fixed-rate: same period and therefore the same work rate as sync,
          but the pest detector is released --phase-frac of a period later, so
          the two windows do not overlap. sync vs phase holds the work constant
          and changes only the overlap.

Outputs (one directory per run)
-------------------------------
  inferences.csv  one row per inference
  system.csv      0.5 s sampler
  power.csv       2 s PMIC sampler (only with --pmic)
  events.csv      SYSTEM_START, RUN_START, THERMAL_CUTOFF, SLOT_SKIPPED, RUN_END, ERROR
  meta.json       configuration, versions, checksums, verification, summary

Usage
-----
  python3 dual_bench.py --check                       # sensors + camera only
  python3 dual_bench.py --policy phase --leaf-model leaf.onnx --pest-model pest.onnx
  python3 dual_bench.py --policy sync --backend fake --allow-no-camera --duration-min 1
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Clock and shared state
# ---------------------------------------------------------------------------

_T0 = time.monotonic()
STOP = threading.Event()


def now():
    """Seconds since the run origin, on the monotonic clock."""
    return time.monotonic() - _T0


def reset_origin():
    global _T0
    _T0 = time.monotonic()


def wall():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Platform probes (all return None when unavailable, never raise)
# ---------------------------------------------------------------------------

def read_float(path, scale=1.0):
    try:
        with open(path) as f:
            return float(f.read().strip()) / scale
    except Exception:
        return None


def read_temp():
    return read_float("/sys/class/thermal/thermal_zone0/temp", 1000.0)


def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True,
                              text=True, timeout=2).stdout.strip()
    except Exception:
        return ""


def arm_clock_mhz():
    """Clock as the firmware reports it. cpufreq shows the requested value,
    which does not reflect firmware-imposed capping."""
    out = vcgencmd("measure_clock", "arm")
    try:
        return int(out.split("=")[1]) / 1e6
    except Exception:
        return None


def throttled():
    """(live bits 0-3, raw word). Only the live nibble is decoded; the upper
    bits latch until reboot (the defect disclosed in the SAT paper)."""
    out = vcgencmd("get_throttled")
    try:
        raw = int(out.split("=")[1].strip(), 16)
        return raw & 0xF, raw
    except Exception:
        return None, None


def find_fan_dir():
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(d, "name")) as f:
                if f.read().strip() == "pwmfan":
                    return d
        except Exception:
            pass
    hits = glob.glob("/sys/devices/platform/cooling_fan/hwmon/hwmon*")
    return hits[0] if hits else None


def find_ambient_sensor():
    hits = glob.glob("/sys/bus/w1/devices/28-*/temperature")
    return hits[0] if hits else None


def parse_pmic(text):
    """Parse `vcgencmd pmic_read_adc`. Returns ({rail: watts}, ext5v_volts).
    Rails reporting both current and voltage are multiplied; voltage-only
    rails (EXT5V, BATT) are not summed, matching the SAT paper's boundary."""
    cur, volt = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        name_part, val = line.split("=", 1)
        name = name_part.split()[0] if name_part.split() else ""
        try:
            v = float(val.strip().rstrip("AV"))
        except ValueError:
            continue
        if name.endswith("_A"):
            cur[name[:-2]] = v
        elif name.endswith("_V"):
            volt[name[:-2]] = v
    rails = {r: cur[r] * volt[r] for r in sorted(cur) if r in volt}
    return rails, volt.get("EXT5V")


def thread_count():
    try:
        return len(os.listdir("/proc/self/task"))
    except Exception:
        return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class CsvLog:
    def __init__(self, path, header):
        self.f = open(path, "w", newline="", buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(header)
        self.lock = threading.Lock()
        self.closed = False

    def row(self, r):
        with self.lock:
            if not self.closed:
                self.w.writerow(r)

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.f.flush()
                self.f.close()


class EventLog(CsvLog):
    def __init__(self, path):
        super().__init__(path, ["t_s", "wall", "event", "temp_c", "note"])
        self.counts = {}

    def log(self, event, note=""):
        t = read_temp()
        self.counts[event] = self.counts.get(event, 0) + 1
        self.row([f"{now():.6f}", wall(), event,
                  "" if t is None else f"{t:.1f}", note])


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def patch_ort_threads(n):
    """Force ONNX Runtime's intra-op pool size for every session created after
    this call. Ultralytics constructs its session internally, so the class is
    wrapped before Ultralytics is imported."""
    import onnxruntime as ort
    base = ort.InferenceSession

    class Patched(base):
        def __init__(self, path_or_bytes, sess_options=None, providers=None,
                     provider_options=None, **kw):
            so = sess_options if sess_options is not None else ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options=so, providers=providers,
                             provider_options=provider_options, **kw)

    ort.InferenceSession = Patched


class YoloBackend:
    def __init__(self, name, path, imgsz, conf, iou):
        from ultralytics import YOLO
        import numpy as np
        self.name, self.imgsz, self.conf, self.iou = name, imgsz, conf, iou
        self.model = YOLO(path, task="detect")
        self.model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                   imgsz=imgsz, verbose=False)

    def ort_intra_threads(self):
        try:
            sess = self.model.predictor.model.session
            return sess.get_session_options().intra_op_num_threads
        except Exception:
            return None

    def infer(self, frame):
        r = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                       verbose=False)
        boxes = r[0].boxes
        n = 0 if boxes is None else len(boxes)
        cmax = float(boxes.conf.max()) if n else 0.0
        self.last_dets = ([] if not n else
                          [(int(c), float(p)) for c, p in zip(boxes.cls.tolist(), boxes.conf.tolist())])
        return n, cmax


class FakeBackend:
    """For testing the scheduler off-device. Work progresses 1.7x slower while
    the other detector is also executing, a crude stand-in for contention."""
    _active = set()
    _lock = threading.Lock()
    BASE = {"leaf": 0.46, "pest": 0.20}

    def __init__(self, name, *_, rng=None):
        self.name = name
        self.rng = rng or random.Random(0)

    def ort_intra_threads(self):
        return None

    def infer(self, frame):
        work = self.BASE[self.name] * math.exp(self.rng.gauss(0, 0.08))
        done, step = 0.0, 0.005
        with FakeBackend._lock:
            FakeBackend._active.add(self.name)
        try:
            while done < work:
                time.sleep(step)
                with FakeBackend._lock:
                    contended = len(FakeBackend._active) > 1
                done += step / (1.7 if contended else 1.0)
        finally:
            with FakeBackend._lock:
                FakeBackend._active.discard(self.name)
        self.last_dets = [(0, 0.5)] if self.rng.random() < 0.5 else []
        return len(self.last_dets), (0.5 if self.last_dets else 0.0)


# ---------------------------------------------------------------------------
# Frame source (one capture thread; detectors read the latest frame)
# ---------------------------------------------------------------------------

class FrameSource(threading.Thread):
    def __init__(self, allow_no_camera, size=640, usb_wh=(640, 480), prefer_usb=False,
                 usb_fps=30):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame, self.fid, self.fts = None, 0, 0.0
        self.count = 0
        self.usb_wh, self.prefer_usb, self.usb_fps = usb_wh, prefer_usb, usb_fps
        self.resolution = None
        self.usb_settings = {}
        self.kind = self._open(allow_no_camera, size)

    def _open(self, allow_no_camera, size):
        picam_error = "skipped (--camera usb)"
        if not self.prefer_usb:
            err = self._open_picamera(size)
            if err is None:
                return "picamera"
            picam_error = err
        try:
            import cv2
            self.cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cam.isOpened():
                self.cam = cv2.VideoCapture(0)
            if self.cam.isOpened():
                self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.usb_wh[0])
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.usb_wh[1])
                self.cam.set(cv2.CAP_PROP_FPS, self.usb_fps)
                self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.usb_settings = self._fix_framerate()
                self.usb_settings["fps_reported"] = self.cam.get(cv2.CAP_PROP_FPS)
                ok, f = self.cam.read()
                if ok:
                    self.resolution = [int(f.shape[1]), int(f.shape[0])]
                    return "usb"
        except Exception:
            pass
        if allow_no_camera:
            import numpy as np
            self.blank = np.zeros((size, size, 3), dtype=np.uint8)
            self.resolution = [size, size]
            print("WARNING: no camera; synthetic frames. "
                  "Telemetry from this run must not be reported.")
            return "none"
        raise SystemExit(
            "FATAL: no camera available.\n"
            f"  picamera2: {picam_error}\n"
            "  cv2.VideoCapture(0): device did not open or returned no frame\n"
            "Aborting rather than benchmarking without a camera "
            "(see the _init_camera docstring in deployment/detection.py).")

    @staticmethod
    def _fix_framerate():
        """UVC webcams (Logitech included) lower the frame rate in dim light by
        lengthening the exposure. Turning off that 'dynamic frame rate' keeps the
        capture load constant. The control has two names across kernels; both
        are tried and the outcome recorded."""
        out = {}
        for ctrl in ("exposure_dynamic_framerate", "exposure_auto_priority"):
            try:
                r = subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c", f"{ctrl}=0"],
                                   capture_output=True, text=True, timeout=3)
                out[ctrl] = "set" if r.returncode == 0 else (r.stderr.strip()[:80] or "failed")
            except Exception as e:
                out[ctrl] = f"{type(e).__name__}"
        return out

    def _open_picamera(self, size):
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_preview_configuration(
                main={"size": (size, size), "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            self.resolution = [size, size]
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def _grab(self):
        if self.kind == "picamera":
            return self.cam.capture_array()
        if self.kind == "usb":
            ok, f = self.cam.read()
            return f if ok else None
        time.sleep(1 / 15)
        return self.blank

    def run(self):
        while not STOP.is_set():
            f = self._grab()
            if f is None:
                continue
            with self.lock:
                self.frame, self.fid, self.fts = f, self.fid + 1, now()
                self.count += 1

    def latest(self):
        while not STOP.is_set():
            with self.lock:
                if self.frame is not None:
                    return self.frame, self.fid, self.fts
            time.sleep(0.01)
        return None, 0, 0.0

    def take_count(self):
        with self.lock:
            c, self.count = self.count, 0
        return c


class ReplaySource(threading.Thread):
    """Replay of still images in place of the camera. Each image is shown for
    `dwell` seconds, then a blank frame for `gap` seconds, in a fixed shuffled
    order (seed) that repeats until the run ends. frame_id counts
    presentations; fts is the onset of the current presentation, so frame_age
    in inferences.csv is the time since the image appeared. A schedule file
    (replay.csv) records every presentation: index, image, onset, offset."""

    def __init__(self, folder, dwell, gap, seed, size, outdir):
        super().__init__(daemon=True)
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.paths = sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                            if p.lower().endswith(exts))
        if not self.paths:
            raise SystemExit(f"FATAL: no images under {folder}")
        self.order = list(range(len(self.paths)))
        random.Random(seed).shuffle(self.order)
        self.dwell, self.gap, self.size = dwell, gap, size
        self.blank = np.full((size, size, 3), 114, dtype=np.uint8)
        self.lock = threading.Lock()
        self.frame, self.fid, self.fts = self.blank, 0, now()
        self.count = 0
        self.kind, self.resolution, self.usb_settings = "replay", [size, size], {}
        self.log = CsvLog(os.path.join(outdir, "replay.csv"),
                          ["presentation", "image", "onset", "offset"])
        self.cache = {}

    def _load(self, i):
        if i not in self.cache:
            im = self.cv2.imread(self.paths[i])
            if im is None:
                im = self.blank
            elif im.shape[:2] != (self.size, self.size):
                im = self.cv2.resize(im, (self.size, self.size), interpolation=self.cv2.INTER_AREA)
            self.cache[i] = im
        return self.cache[i]

    def run(self):
        k = 0
        pres = 0
        while not STOP.is_set():
            i = self.order[k % len(self.order)]
            k += 1
            pres += 1
            im = self._load(i)
            with self.lock:
                self.frame, self.fid, self.fts = im, pres, now()
                self.count += 1
            onset = self.fts
            if STOP.wait(self.dwell):
                break
            off = now()
            self.log.row([pres, os.path.basename(self.paths[i]), f"{onset:.6f}", f"{off:.6f}"])
            with self.lock:
                self.frame, self.fid, self.fts = self.blank, 0, off
                self.count += 1
            if STOP.wait(self.gap):
                break

    def latest(self):
        with self.lock:
            return self.frame, self.fid, self.fts

    def take_count(self):
        with self.lock:
            c, self.count = self.count, 0
        return c


# ---------------------------------------------------------------------------
# Software thermal cut-off (shared by both detectors)
# ---------------------------------------------------------------------------

class CutoffGate:
    """Same semantics as detection.py: before an inference starts, if the die
    is above the limit, record THERMAL_CUTOFF, suspend for a fixed pause, then
    re-test. No hysteresis. One gate is shared, so a cut-off suspends both
    detectors and each event costs exactly `pause` seconds of monitoring."""

    def __init__(self, limit, pause, events):
        self.limit, self.pause, self.events = limit, pause, events
        self.until = 0.0
        self.lock = threading.Lock()

    def wait(self, who):
        while not STOP.is_set():
            with self.lock:
                t = now()
                if t < self.until:
                    w = self.until - t
                else:
                    temp = read_temp()
                    if temp is not None and temp > self.limit:
                        self.until = t + self.pause
                        self.events.log("THERMAL_CUTOFF", f"by={who}")
                        w = self.pause
                    else:
                        return True
            STOP.wait(w)
        return False


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class SystemSampler(threading.Thread):
    HEADER = ["t_s", "wall", "phase", "temp_c", "arm_mhz", "cpufreq_mhz",
              "throttled_now", "throttled_raw", "fan_rpm", "fan_pwm",
              "cpu_pct", "ram_pct", "ambient_c", "capture_fps", "threads"]

    def __init__(self, path, interval, source, state):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        psutil.cpu_percent(interval=None)
        self.log = CsvLog(path, self.HEADER)
        self.interval, self.source, self.state = interval, source, state
        self.fan = find_fan_dir()
        self.ambient = find_ambient_sensor()

    def run(self):
        nxt = now()
        last = now()
        while not STOP.is_set():
            nxt += self.interval
            live, raw = throttled()
            t = now()
            fps = self.source.take_count() / max(t - last, 1e-6)
            last = t
            fan_rpm = read_float(os.path.join(self.fan, "fan1_input")) if self.fan else None
            fan_pwm = read_float(os.path.join(self.fan, "pwm1")) if self.fan else None
            amb = read_float(self.ambient, 1000.0) if self.ambient else None
            cf = read_float("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000.0)
            temp = read_temp()
            arm = arm_clock_mhz()
            self.log.row([
                f"{t:.6f}", wall(), self.state["phase"],
                "" if temp is None else f"{temp:.2f}",
                "" if arm is None else f"{arm:.0f}",
                "" if cf is None else f"{cf:.0f}",
                "" if live is None else hex(live),
                "" if raw is None else hex(raw),
                "" if fan_rpm is None else f"{fan_rpm:.0f}",
                "" if fan_pwm is None else f"{fan_pwm:.0f}",
                f"{self.psutil.cpu_percent(interval=None):.1f}",
                f"{self.psutil.virtual_memory().percent:.1f}",
                "" if amb is None else f"{amb:.2f}",
                f"{fps:.2f}", thread_count()])
            STOP.wait(max(0.0, nxt - now()))
        self.log.close()


class PowerSampler(threading.Thread):
    def __init__(self, path, interval, state, jitter=0.0, seed=0):
        super().__init__(daemon=True)
        # jitter > 0 adds U(0, jitter) seconds to every interval so that the
        # sampler cannot lock onto a fixed-period release schedule (campaign1
        # used jitter = 0, and the fixed-period conditions were phase-locked).
        self.jitter, self.rng = jitter, random.Random(seed)
        rails, _ = parse_pmic(vcgencmd("pmic_read_adc"))
        self.rails = list(rails)
        if not self.rails:
            raise SystemExit("FATAL: --pmic given but `vcgencmd pmic_read_adc` "
                             "returned no rails.")
        self.log = CsvLog(path, ["t_s", "wall", "phase", "total_w", "ext5v_v"]
                          + [f"{r}_W" for r in self.rails])
        self.interval, self.state = interval, state

    def run(self):
        nxt = now()
        while not STOP.is_set():
            nxt += self.interval + (self.rng.uniform(0.0, self.jitter) if self.jitter else 0.0)
            rails, ext5 = parse_pmic(vcgencmd("pmic_read_adc"))
            self.log.row([f"{now():.6f}", wall(), self.state["phase"],
                          f"{sum(rails.values()):.4f}",
                          "" if ext5 is None else f"{ext5:.4f}"]
                         + [f"{rails.get(r, float('nan')):.4f}" for r in self.rails])
            STOP.wait(max(0.0, nxt - now()))
        self.log.close()


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

INF_HEADER = ["model", "idx", "t_start", "t_end", "latency_ms", "frame_id",
              "frame_age_ms", "t_sched", "late_ms", "n_boxes", "conf_max", "dets"]


class Bench:
    def __init__(self, a, outdir):
        self.a, self.outdir = a, outdir
        self.state = {"phase": "init"}
        self.events = EventLog(os.path.join(outdir, "events.csv"))
        self.inf = CsvLog(os.path.join(outdir, "inferences.csv"), INF_HEADER)
        self.counters = {"leaf": 0, "pest": 0}
        self.skipped = {"leaf": 0, "pest": 0, "sync": 0}
        self.errors = []
        self.gate = CutoffGate(a.cutoff, a.cutoff_pause, self.events)

    # -- one inference -----------------------------------------------------
    def run_one(self, key, sched=None):
        if not self.gate.wait(key):
            return False
        frame, fid, fts = self.source.latest()
        if frame is None:
            return False
        t_s = now()
        n, cmax = self.backends[key].infer(frame)
        t_e = now()
        self.counters[key] += 1
        self.inf.row([key, self.counters[key], f"{t_s:.6f}", f"{t_e:.6f}",
                      f"{(t_e - t_s) * 1000:.2f}", fid,
                      f"{(t_s - fts) * 1000:.1f}",
                      "" if sched is None else f"{sched:.6f}",
                      "" if sched is None else f"{(t_s - sched) * 1000:.2f}",
                      n, f"{cmax:.3f}",
                      ";".join(f"{c}:{p:.2f}" for c, p in getattr(self.backends[key], "last_dets", []))])
        return True

    def running(self):
        return not STOP.is_set() and now() < self.end

    # -- policies ----------------------------------------------------------
    def p_indep(self, key, sleep_s, delay):
        if delay > 0 and STOP.wait(delay):
            return
        while self.running():
            if not self.run_one(key):
                break
            STOP.wait(sleep_s)

    def p_serial(self):
        while self.running():
            if not self.run_one("leaf") or not self.run_one("pest"):
                break
            STOP.wait(self.a.serial_sleep)

    def p_fixed(self, key, period, phase_s, skip_key=None):
        k = 0
        while True:
            sched = self.start + phase_s + k * period
            if sched >= self.end or STOP.is_set():
                break
            d = sched - now()
            if d > 0 and STOP.wait(d):
                break
            if not self.run_one(key, sched):
                break
            nk = k + 1
            t = now()
            if self.start + phase_s + nk * period < t:
                nk = math.ceil((t - self.start - phase_s) / period)
                self.skipped[skip_key or key] += nk - k - 1
                self.events.log("SLOT_SKIPPED", f"{key} skipped={nk - k - 1}")
            k = nk

    def p_serialp(self, period):
        """Fixed-period serial: leaf at each release, pest immediately after it.
        Both rows carry the release instant as t_sched, so time-to-both is measured
        from the release as in sync/phase."""
        k = 0
        while True:
            sched = self.start + k * period
            if sched >= self.end or STOP.is_set():
                break
            d = sched - now()
            if d > 0 and STOP.wait(d):
                break
            if not self.run_one("leaf", sched) or not self.run_one("pest", sched):
                break
            nk = k + 1
            t = now()
            if self.start + nk * period < t:
                nk = math.ceil((t - self.start) / period)
                self.skipped["sync"] += nk - k - 1
                self.events.log("SLOT_SKIPPED", f"serialp skipped={nk - k - 1}")
            k = nk

    def p_sync(self, period):
        go = {k: threading.Event() for k in ("leaf", "pest")}
        done = {k: threading.Event() for k in ("leaf", "pest")}
        sched_box = {"t": None}

        def worker(key):
            while True:
                go[key].wait()
                go[key].clear()
                if STOP.is_set() or sched_box["t"] is None:
                    done[key].set()
                    return
                self.run_one(key, sched_box["t"])
                done[key].set()

        ws = [threading.Thread(target=self._guard, args=(worker, k), daemon=True)
              for k in ("leaf", "pest")]
        for w in ws:
            w.start()
        k = 0
        while True:
            sched = self.start + k * period
            if sched >= self.end or STOP.is_set():
                break
            d = sched - now()
            if d > 0 and STOP.wait(d):
                break
            if not self.gate.wait("sync"):
                break
            sched_box["t"] = sched
            for key in go:
                done[key].clear()
                go[key].set()
            for key in done:
                while not done[key].wait(0.5):
                    if STOP.is_set():
                        break
            nk = k + 1
            t = now()
            if self.start + nk * period < t:
                nk = math.ceil((t - self.start) / period)
                self.skipped["sync"] += nk - k - 1
                self.events.log("SLOT_SKIPPED", f"sync skipped={nk - k - 1}")
            k = nk
        sched_box["t"] = None
        for key in go:
            go[key].set()
        for w in ws:
            w.join(timeout=10)

    # -- harness -----------------------------------------------------------
    def _guard(self, fn, *args):
        try:
            fn(*args)
        except Exception:
            tb = traceback.format_exc()
            self.errors.append(tb)
            self.events.log("ERROR", tb.strip().splitlines()[-1][:200])
            print(tb, file=sys.stderr)
            STOP.set()

    def threads_for_policy(self):
        a = self.a
        T = threading.Thread
        g = self._guard
        if a.policy == "indep":
            return [T(target=g, args=(self.p_indep, "leaf", a.leaf_sleep, 0.0)),
                    T(target=g, args=(self.p_indep, "pest", a.pest_sleep, a.start_offset))]
        if a.policy == "serial":
            return [T(target=g, args=(self.p_serial,))]
        if a.policy == "sync":
            return [T(target=g, args=(self.p_sync, a.period))]
        if a.policy == "serialp":
            return [T(target=g, args=(self.p_serialp, a.period))]
        if a.policy == "phase":
            offset = a.phase_s if a.phase_s is not None else a.period * a.phase_frac
            return [T(target=g, args=(self.p_fixed, "leaf", a.period, 0.0)),
                    T(target=g, args=(self.p_fixed, "pest", a.period, offset))]
        raise ValueError(a.policy)

    def run(self, meta):
        a = self.a
        meta["threads_before_models"] = thread_count()
        if a.ort_threads > 0 and a.backend == "yolo":
            patch_ort_threads(a.ort_threads)
        rng = random.Random(a.seed)
        if a.backend == "yolo":
            self.backends = {
                "leaf": YoloBackend("leaf", a.leaf_model, a.imgsz, a.conf, a.iou),
                "pest": YoloBackend("pest", a.pest_model, a.imgsz, a.conf, a.iou)}
        else:
            self.backends = {k: FakeBackend(k, rng=random.Random(rng.random()))
                             for k in ("leaf", "pest")}
        meta["threads_after_models"] = thread_count()
        meta["ort_intra_op_reported"] = {k: b.ort_intra_threads()
                                         for k, b in self.backends.items()}
        if a.ort_threads > 0 and a.backend == "yolo":
            rep = meta["ort_intra_op_reported"]
            if any(v not in (None, a.ort_threads) for v in rep.values()):
                raise SystemExit(f"FATAL: requested {a.ort_threads} ORT threads, "
                                 f"sessions report {rep}.")

        if a.replay:
            self.source = ReplaySource(a.replay, a.dwell, a.gap, a.seed, a.imgsz, self.outdir)
            meta["replay"] = {"folder": os.path.abspath(a.replay), "images": len(self.source.paths),
                              "dwell_s": a.dwell, "gap_s": a.gap, "seed": a.seed}
        else:
            self.source = FrameSource(a.allow_no_camera, a.imgsz,
                                      (a.usb_width, a.usb_height), a.camera == "usb", a.usb_fps)
        meta["camera"] = self.source.kind
        meta["camera_resolution"] = self.source.resolution
        meta["usb_settings"] = self.source.usb_settings
        if self.source.kind == "usb" and self.source.resolution != [a.usb_width, a.usb_height]:
            print(f"WARNING: requested {a.usb_width}x{a.usb_height}, camera gives "
                  f"{self.source.resolution}; recorded in meta.json")
        if a.backend == "yolo" and self.source.kind == "none" and not a.replay:
            raise SystemExit("FATAL: real backend without a camera is not allowed.")

        reset_origin()
        meta["origin_wall"] = wall()
        self.source.start()
        sampler = SystemSampler(os.path.join(self.outdir, "system.csv"),
                                a.sample_s, self.source, self.state)
        meta["fan_hwmon"] = sampler.fan
        meta["ambient_sensor"] = sampler.ambient
        samplers = [sampler]
        if a.pmic:
            samplers.append(PowerSampler(os.path.join(self.outdir, "power.csv"),
                                         a.pmic_s, self.state, a.pmic_jitter, a.seed))
        for s in samplers:
            s.start()

        self.events.log("SYSTEM_START", f"policy={a.policy} label={a.label}")
        self.state["phase"] = "idle"
        if STOP.wait(a.idle_s):
            return
        self.state["phase"] = "run"
        self.start = now()
        self.end = self.start + a.duration_min * 60.0
        meta["run_start_s"] = self.start
        meta["run_start_wall"] = wall()
        self.events.log("RUN_START", f"policy={a.policy}")
        ts = self.threads_for_policy()
        for t in ts:
            t.start()
        for t in ts:
            while t.is_alive():
                t.join(timeout=1.0)
        meta["run_end_s"] = now()
        self.state["phase"] = "post"
        self.events.log("RUN_END", f"leaf={self.counters['leaf']} pest={self.counters['pest']}")


def build_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", choices=["indep", "serial", "sync", "phase", "serialp"])
    p.add_argument("--check", action="store_true",
                   help="probe sensors, PMIC and camera, then exit")
    p.add_argument("--label", default=None)
    p.add_argument("--outdir", default=os.path.join(here, "runs"))
    p.add_argument("--backend", choices=["yolo", "fake"], default="yolo")
    p.add_argument("--allow-no-camera", action="store_true",
                   help="fake backend only; such runs must never be reported")
    p.add_argument("--leaf-model", default=os.path.join(here, "leaf.onnx"))
    p.add_argument("--pest-model", default=os.path.join(here, "pest.onnx"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--camera", choices=["auto", "usb"], default="auto",
                   help="auto: CSI camera first, then USB; usb: USB only")
    p.add_argument("--usb-width", type=int, default=640)
    p.add_argument("--usb-height", type=int, default=480)
    p.add_argument("--usb-fps", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--duration-min", type=float, default=90.0)
    p.add_argument("--idle-s", type=float, default=60.0)
    p.add_argument("--leaf-sleep", type=float, default=0.8)
    p.add_argument("--pest-sleep", type=float, default=1.2)
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--serial-sleep", type=float, default=1.2)
    p.add_argument("--period", type=float, default=1.5)
    p.add_argument("--phase-frac", type=float, default=0.5)
    p.add_argument("--phase-s", type=float, default=None,
                   help="pest release offset in seconds (overrides --phase-frac)")
    p.add_argument("--replay", default=None,
                   help="folder of still images replayed in place of the camera (see ReplaySource)")
    p.add_argument("--dwell", type=float, default=2.5, help="seconds each replayed image is shown")
    p.add_argument("--gap", type=float, default=1.0, help="seconds of blank frame between images")
    p.add_argument("--ort-threads", type=int, default=0,
                   help="ONNX Runtime intra-op threads per session (0 = ORT default)")
    p.add_argument("--cutoff", type=float, default=82.0)
    p.add_argument("--cutoff-pause", type=float, default=5.0)
    p.add_argument("--sample-s", type=float, default=0.5)
    p.add_argument("--pmic", action="store_true")
    p.add_argument("--pmic-s", type=float, default=2.0)
    p.add_argument("--pmic-jitter", type=float, default=0.0,
                   help="random extra delay (s, uniform) added to each PMIC interval; "
                        "use ~0.7 with fixed-period policies (sync/phase), see README")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if not a.check and not a.policy:
        p.error("--policy is required unless --check is given")
    if a.allow_no_camera and a.backend != "fake":
        p.error("--allow-no-camera is only permitted with --backend fake")
    return a


def check():
    print("die temperature  :", read_temp())
    print("ARM clock (MHz)  :", arm_clock_mhz())
    print("throttled        :", throttled())
    fan = find_fan_dir()
    print("fan hwmon        :", fan)
    if fan:
        print("  fan1_input rpm :", read_float(os.path.join(fan, "fan1_input")))
        print("  pwm1           :", read_float(os.path.join(fan, "pwm1")))
    amb = find_ambient_sensor()
    print("ambient DS18B20  :", amb, read_float(amb, 1000.0) if amb else None)
    rails, ext5 = parse_pmic(vcgencmd("pmic_read_adc"))
    print(f"PMIC rails       : {len(rails)} summed, total {sum(rails.values()):.3f} W, EXT5V {ext5}")
    print("governor         :", read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
    src = FrameSource(False, prefer_usb=True)
    print("camera           :", src.kind, src.resolution, src.usb_settings)
    if src.kind != "none":
        src.start()
        time.sleep(1.0)
        src.take_count()
        time.sleep(5.0)
        print(f"capture rate     : {src.take_count() / 5.0:.1f} fps over 5 s")
        STOP.set()


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine(), "node": platform.node()}
    # importlib.metadata, not import: importing ultralytics here would happen
    # before patch_ort_threads() and could defeat the thread-count control.
    from importlib import metadata
    for dist in ("onnxruntime", "ultralytics", "numpy", "opencv-python",
                 "opencv-python-headless", "psutil", "picamera2"):
        try:
            v[dist] = metadata.version(dist)
        except Exception:
            v[dist] = None
    v["firmware"] = vcgencmd("version") or None
    v["governor"] = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    v["model"] = read_text("/proc/device-tree/model")
    return v


def main(argv=None):
    a = build_args(argv)
    if a.check:
        check()
        return 0
    label = a.label or f"{a.policy}_{datetime.now():%Y%m%d_%H%M%S}"
    a.label = label
    outdir = os.path.join(a.outdir, label)
    os.makedirs(outdir, exist_ok=False)
    meta = {"label": label, "args": vars(a), "versions": versions(),
            "status": "started", "start_wall": wall()}
    if a.backend == "yolo":
        meta["model_md5"] = {"leaf": md5(a.leaf_model), "pest": md5(a.pest_model)}
    meta_path = os.path.join(outdir, "meta.json")

    def dump():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    dump()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    bench = Bench(a, outdir)
    status = "ok"
    try:
        bench.run(meta)
    except KeyboardInterrupt:
        status = "interrupted"
    except SystemExit as e:
        status = f"aborted: {e}"
        print(e, file=sys.stderr)
    except Exception:
        status = "error"
        meta["traceback"] = traceback.format_exc()
        print(meta["traceback"], file=sys.stderr)
    finally:
        if STOP.is_set() and status == "ok":
            status = "error" if bench.errors else "interrupted"
        STOP.set()
        time.sleep(max(a.sample_s, 0.2) + 0.3)
        if bench.errors:
            meta["worker_errors"] = bench.errors
        meta.update({"status": status, "end_wall": wall(),
                     "inferences": dict(bench.counters),
                     "slots_skipped": dict(bench.skipped),
                     "events": dict(bench.events.counts)})
        bench.inf.close()
        bench.events.close()
        dump()
    print(f"{label}: {status}  leaf={bench.counters['leaf']} pest={bench.counters['pest']}"
          f"  cutoffs={bench.events.counts.get('THERMAL_CUTOFF', 0)}  -> {outdir}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
