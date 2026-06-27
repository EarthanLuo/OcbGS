#!/usr/bin/env python3
"""Plot the Exp A controllable-budget Pareto front (demand vs uniform).

Reads the CSV from `collect_results.py pareto` and draws one curve per arm:
x = achieved anchors, y = the chosen quality metric. Demand on/above uniform
(especially at the low-budget / left end) is the Secondary claim.
"""
import argparse
import csv as csv_mod


def load_pareto_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv_mod.DictReader(f):
            rows.append({
                "arm": r["arm"],
                "factor": float(r["factor"]),
                "anchors": float(r["anchors"]),
                "PSNR": float(r["PSNR"]),
                "SSIM": float(r["SSIM"]),
                "LPIPS": float(r["LPIPS"]),
                "n": int(r["n"]),
            })
    return rows


def split_curves(rows, metric):
    curves = {}
    for r in rows:
        curves.setdefault(r["arm"], []).append((r["anchors"], r[metric]))
    out = {}
    for arm, pts in curves.items():
        pts.sort(key=lambda p: p[0])
        out[arm] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="PSNR", choices=["PSNR", "SSIM", "LPIPS"])
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Controllable-budget Pareto — amsterdam")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_pareto_rows(args.csv)
    curves = split_curves(rows, args.metric)

    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in sorted(curves):
        xs, ys = curves[arm]
        ax.plot(xs, ys, marker="o", label=arm)
    ax.set_xlabel("achieved anchors")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
