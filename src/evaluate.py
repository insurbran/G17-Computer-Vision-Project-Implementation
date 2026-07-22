"""MAE / MAPE evaluation against hand-built ground truth.

Pass one or more estimate CSVs to compare methods side by side, e.g.:
  python src/evaluate.py --ground-truth data/ground_truth.csv \
      --estimates outputs/estimates_box.csv outputs/estimates_segment.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def score(est_path: str, gt: pd.DataFrame) -> dict:
    est = pd.read_csv(est_path)
    m = gt.merge(est, on="file", how="inner")
    if m.empty:
        raise SystemExit(f"No overlapping files between ground truth and {est_path}")
    err = m["estimated_kcal"] - m["expected_kcal"]
    mae = err.abs().mean()
    mape = (err.abs() / m["expected_kcal"].clip(lower=1)).mean() * 100
    return {"name": Path(est_path).stem, "n": len(m), "MAE_kcal": round(mae, 1),
            "MAPE_%": round(mape, 1), "bias_kcal": round(err.mean(), 1), "merged": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--estimates", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="optional CSV for per-image errors")
    args = ap.parse_args()

    gt = pd.read_csv(args.ground_truth)[["file", "expected_kcal"]].dropna()
    results = [score(p, gt) for p in args.estimates]

    print(f"\n{'method':<28}{'n':>4}{'MAE (kcal)':>12}{'MAPE (%)':>10}{'bias':>8}")
    for r in results:
        print(f"{r['name']:<28}{r['n']:>4}{r['MAE_kcal']:>12}{r['MAPE_%']:>10}{r['bias_kcal']:>8}")

    worst = results[-1]["merged"].assign(
        abs_err=lambda d: (d["estimated_kcal"] - d["expected_kcal"]).abs()
    ).nlargest(5, "abs_err")[["file", "expected_kcal", "estimated_kcal", "abs_err"]]
    print("\nWorst 5 images (last method) — analyse these in the report:")
    print(worst.to_string(index=False))

    if args.out:
        results[-1]["merged"].to_csv(args.out, index=False)
        print(f"\nPer-image errors -> {args.out}")


if __name__ == "__main__":
    main()
