"""
aggregate_runs.py  --  CSC3014 Project Part 2, Track A (G17)

Evaluates every trained run on the held-out test split and builds the
baseline vs CLAHE comparison table with per-condition mean and spread.

The point of the spread column: a difference between two conditions only means
something if it is larger than the variation you get from re-rolling the seed
within a single condition. This script puts both numbers side by side so that
comparison is honest rather than assumed.

CLAHE models are evaluated on the CLAHE test images, baseline models on the
raw test images. Evaluating a model on preprocessing it never saw would measure
a mismatch, not the method.

Usage (from C:\\Users\\PF4B3\\Desktop\\g17):
    python aggregate_runs.py
    python aggregate_runs.py --skip-eval      (reuse cached results)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path("C:/Users/PF4B3/runs/detect")
CACHE = HERE / "test_results.json"

# run name -> which data.yaml it must be tested against
CONDITIONS = {
    "baseline": {
        "yaml": HERE / "dataset" / "data.yaml",
        "runs": ["baseline3", "baseline_s1", "baseline_s2"],
        "seeds": [42, 1, 2],
    },
    "clahe": {
        "yaml": HERE / "dataset_clahe" / "data.yaml",
        "runs": ["clahe", "clahe_s1", "clahe_s2"],
        "seeds": [42, 1, 2],
    },
}

METRICS = ("mAP50", "mAP50-95", "precision", "recall")


def evaluate(run_name, yaml_path):
    from ultralytics import YOLO

    weights = RUNS / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"  MISSING: {weights}")
        return None

    model = YOLO(str(weights))
    m = model.val(data=str(yaml_path), split="test", verbose=False, plots=False).box

    return {
        "mAP50": float(m.map50),
        "mAP50-95": float(m.map),
        "precision": float(m.mp),
        "recall": float(m.mr),
        "per_class_map50": {model.names[i]: float(v) for i, v in zip(m.ap_class_index, m.ap50)},
    }


def spread(values):
    """Range and sample stdev. Range is the more honest number at n=3."""
    return max(values) - min(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true", help="reuse test_results.json")
    args = parser.parse_args()

    if args.skip_eval and CACHE.exists():
        results = json.loads(CACHE.read_text())
        print(f"reusing {CACHE}\n")
    else:
        results = {}
        for condition, spec in CONDITIONS.items():
            if not spec["yaml"].exists():
                sys.exit(f"missing yaml: {spec['yaml']}")
            for run_name in spec["runs"]:
                print(f"evaluating {run_name} on {spec['yaml'].parent.name}/test ...")
                got = evaluate(run_name, spec["yaml"])
                if got:
                    got["condition"] = condition
                    results[run_name] = got
        CACHE.write_text(json.dumps(results, indent=2))
        print(f"\ncached to {CACHE}\n")

    # ---- per run table ----
    print("=" * 74)
    print("TEST SET, PER RUN")
    print("=" * 74)
    print(f"{'run':<14}{'seed':>6}{'mAP50':>10}{'mAP50-95':>11}{'P':>10}{'R':>10}")
    print("-" * 74)
    for condition, spec in CONDITIONS.items():
        for run_name, seed in zip(spec["runs"], spec["seeds"]):
            if run_name not in results:
                continue
            r = results[run_name]
            print(f"{run_name:<14}{seed:>6}{r['mAP50']:>10.4f}{r['mAP50-95']:>11.4f}"
                  f"{r['precision']:>10.4f}{r['recall']:>10.4f}")
        print()

    # ---- condition summary ----
    print("=" * 74)
    print("CONDITION SUMMARY  (n=3 seeds)")
    print("=" * 74)
    print(f"{'condition':<12}{'metric':<12}{'mean':>10}{'range':>10}{'stdev':>10}"
          f"{'min':>10}{'max':>10}")
    print("-" * 74)

    summary = {}
    for condition, spec in CONDITIONS.items():
        rows = [results[r] for r in spec["runs"] if r in results]
        if not rows:
            continue
        summary[condition] = {}
        for metric in METRICS:
            vals = [r[metric] for r in rows]
            rng, sd = spread(vals)
            summary[condition][metric] = (statistics.mean(vals), rng, sd)
            print(f"{condition:<12}{metric:<12}{statistics.mean(vals):>10.4f}{rng:>10.4f}"
                  f"{sd:>10.4f}{min(vals):>10.4f}{max(vals):>10.4f}")
        print()

    # ---- the verdict ----
    if len(summary) == 2:
        print("=" * 74)
        print("EFFECT vs NOISE")
        print("=" * 74)
        print(f"{'metric':<12}{'baseline':>11}{'clahe':>11}{'diff':>10}"
              f"{'noise':>10}{'verdict':>18}")
        print("-" * 74)
        for metric in METRICS:
            b_mean, b_rng, _ = summary["baseline"][metric]
            c_mean, c_rng, _ = summary["clahe"][metric]
            diff = c_mean - b_mean
            noise = max(b_rng, c_rng)
            verdict = "REAL" if abs(diff) > noise else "within noise"
            print(f"{metric:<12}{b_mean:>11.4f}{c_mean:>11.4f}{diff:>+10.4f}"
                  f"{noise:>10.4f}{verdict:>18}")
        print()
        print("noise = the larger of the two within-condition ranges.")
        print("A difference smaller than that cannot be attributed to CLAHE.")


if __name__ == "__main__":
    main()
