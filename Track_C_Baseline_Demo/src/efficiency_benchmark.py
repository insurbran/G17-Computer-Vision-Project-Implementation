"""
efficiency_benchmark.py -- Track C (G17)

Measures inference cost for both models: parameter count, on-disk size, latency
per image, and throughput.

WHY THIS SCRIPT EXISTS -- it closes a third commitment from our literature review
--------------------------------------------------------------------------------
Section 8.5 of the review ("Computational Efficiency and Real-World Deployment")
names deployment cost as one of six unresolved research gaps:

    "real-world deployment of food image recognition and calorie estimation into
     mobile applications can pose another problem. Calorie estimation requires
     large and complex models for immediate feedback, but those models may not
     perform efficiently in situations where resources are scarce, like mobile
     devices [18]."

Section 9.3's fifth recommendation is blunter:

    "we should start designing mobiles from the beginning. This is because a
     calorie tracker that runs on the phone, which is the most convenient, will
     not be used if it runs slowly."

Track C previously compared the two models on parameter count alone. Parameters
are a proxy for size, not for speed -- a model with fewer parameters can easily be
slower if its operations are less cache-friendly or poorly parallelised.
EfficientNetB0 is a well-known example: its depthwise separable convolutions give
it a low parameter count but a higher latency per parameter than the number
suggests. Reporting parameters and calling it an efficiency analysis would be
exactly the "metric that measures something merely adjacent to the task" that
Section 7.3 warns against, citing Lee [25].

So this measures wall-clock latency directly.

WHAT THE NUMBERS MEAN FOR THE ARGUMENT
--------------------------------------
Track C's finding is that detection beats single-label classification on
multi-item plates. A fair objection is "yes, but the classifier is cheaper, so
maybe it is the right trade on a phone." This script answers that objection with
measurements rather than assertion.

Note the device: latency is hardware-specific and a figure measured on one laptop
does not transfer. The script records the device and thread count so the number
in the report is interpretable.

Usage:
    python src/efficiency_benchmark.py
    python src/efficiency_benchmark.py --runs 50 --device cpu
"""

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from dataset import pick_device


def megabytes(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def time_it(fn, runs: int, warmup: int = 3):
    """-> (mean_ms, stdev_ms, p50_ms). Warm-up runs are discarded.

    The first few inferences of any model are unrepresentative: lazy kernel
    selection, memory allocation and cache population all happen once. Timing
    them would inflate the figure and make the comparison meaningless.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return (round(statistics.mean(samples), 2),
            round(statistics.stdev(samples), 2) if len(samples) > 1 else 0.0,
            round(statistics.median(samples), 2))


def bench_classifier(device, runs):
    from train_classifier import build_model
    cfg.require(cfg.CLASSIFIER_CKPT, "the trained classifier",
                "run: python src/train_classifier.py")
    ckpt = torch.load(cfg.CLASSIFIER_CKPT, map_location=device, weights_only=False)
    model = build_model(cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    x = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE, device=device)

    def run():
        with torch.no_grad():
            model(x)

    mean, sd, p50 = time_it(run, runs)
    return {
        "model": "EfficientNetB0 (classifier)",
        "params": sum(p.numel() for p in model.parameters()),
        "file_mb": megabytes(cfg.CLASSIFIER_CKPT),
        "input": f"{cfg.IMG_SIZE}x{cfg.IMG_SIZE}",
        "latency_mean_ms": mean,
        "latency_stdev_ms": sd,
        "latency_median_ms": p50,
        "throughput_img_per_s": round(1000.0 / mean, 1) if mean else 0.0,
    }


def bench_detector(device, runs):
    from ultralytics import YOLO
    cfg.require_weights()
    model = YOLO(str(cfg.WEIGHTS))
    # A real BGR frame, not a tensor: this is the path app.py actually uses, so
    # the number includes the pre-processing the demo really pays for.
    frame = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

    def run():
        model.predict(frame, conf=cfg.CONF_THRESHOLD, verbose=False, device=str(device))

    mean, sd, p50 = time_it(run, runs)
    n_params = sum(p.numel() for p in model.model.parameters())
    return {
        "model": "YOLOv8n (detector)",
        "params": n_params,
        "file_mb": megabytes(cfg.WEIGHTS),
        "input": "640x640",
        "latency_mean_ms": mean,
        "latency_stdev_ms": sd,
        "latency_median_ms": p50,
        "throughput_img_per_s": round(1000.0 / mean, 1) if mean else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    env = {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "runs": args.runs,
    }

    print("=" * 78)
    print("INFERENCE COST  (review sections 8.5 and 9.3, recommendation 5)")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:<16} {v}")
    print()

    rows = [bench_classifier(device, args.runs), bench_detector(device, args.runs)]

    print(f"{'model':<30}{'params':>11}{'file MB':>9}{'ms/img':>10}{'img/s':>8}")
    for r in rows:
        print(f"{r['model']:<30}{r['params']:>11,}{r['file_mb']:>9}"
              f"{r['latency_mean_ms']:>10}{r['throughput_img_per_s']:>8}")

    clf, det = rows[0], rows[1]
    ratio = det["latency_mean_ms"] / clf["latency_mean_ms"] if clf["latency_mean_ms"] else 0
    print()
    print(f"  The detector costs {ratio:.2f}x the classifier's latency per image.")
    print("  Read alongside M3: on multi-item plates the classifier's calorie error")
    print("  is 5.5x the detector's. Whether that latency is worth paying is the")
    print("  real deployment question, and now it is a measured trade-off rather")
    print("  than an assumed one.")
    print()
    print("  CAVEAT: latency is hardware-specific. Quote the device above with any")
    print("  figure taken from here, and do not compare it against a number from a")
    print("  different machine.")

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_csv = cfg.OUTPUTS / "efficiency_benchmark.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (cfg.OUTPUTS / "efficiency_env.json").write_text(json.dumps(env, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote {out_csv}")
    print(f"wrote {cfg.OUTPUTS / 'efficiency_env.json'}")


if __name__ == "__main__":
    main()
