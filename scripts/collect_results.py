#!/usr/bin/env python3
"""Exp 4 result collector.

Subcommands:
  total_points   Extract per-seed total_points from TB events @ given step(s).
                 Optionally aggregate to mean/stdev and write a BTOTAL file.
  metrics        Collect PSNR/SSIM/LPIPS from results.json across seeds
                 at each checkpoint. Output per-seed table + aggregate summary.
  compare        Read two per-arm summaries + sigma → decision table
                 (|ΔPSNR| vs 2σ at each checkpoint).

Usage examples:
  python scripts/collect_results.py total_points \\
      --glob "/tmp/exp4/garden_baseline/arm_a/seed_*" \\
      --step 25000 --tag-suffix "total_points" \\
      --aggregate mean --output-btotal BTOTAL_GARDEN

  python scripts/collect_results.py metrics \\
      --glob "/tmp/exp4/garden_baseline/arm_b/seed_*" \\
      --output /tmp/exp4/sigma_garden.json

  python scripts/collect_results.py compare \\
      --a-only /tmp/exp4/sigma_garden.json \\
      --a-plus-b /tmp/exp4/summary_a_plus_b.json \\
      --sigma /tmp/exp4/sigma_garden.json
"""

import argparse
import glob as glob_mod
import json
import os
import statistics
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_events_dir(seed_dir):
    """Locate the directory containing events.out.tfevents.* files."""
    for root, dirs, files in os.walk(seed_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return root
    return None


def _find_total_points_tag(tags):
    """Find the total_points scalar tag among TB tags (suffix match)."""
    for tag in tags:
        if tag.endswith("/total_points"):
            return tag
    return None


def _read_total_points_at_step(events_dir, tag, step):
    ea = EventAccumulator(events_dir)
    ea.Reload()
    scalars = ea.Scalars(tag)
    values = []
    for event in scalars:
        if step is None or event.step == step:
            values.append(event.value)
    if not values:
        return None
    # If step specified, return the exact match; otherwise latest
    return values[-1]


def _read_results_json(results_path):
    with open(results_path, 'r') as f:
        return json.load(f)


def _seed_dirs(pattern):
    return sorted(glob_mod.glob(pattern))


def _seed_name(seed_dir):
    return os.path.basename(os.path.normpath(seed_dir))


# ---------------------------------------------------------------------------
# Subcommand: total_points
# ---------------------------------------------------------------------------

def cmd_total_points(args):
    if not HAS_TB:
        print("ERROR: tensorboard not available. pip install tensorboard")
        sys.exit(1)

    seeds = _seed_dirs(args.glob)
    if not seeds:
        print(f"ERROR: no seed dirs match {args.glob}")
        sys.exit(1)

    table = []  # list of {seed, step, value}
    for sd in seeds:
        events_dir = _find_events_dir(sd)
        if events_dir is None:
            print(f"WARN: no events found in {sd}")
            continue
        ea = EventAccumulator(events_dir)
        ea.Reload()
        tag = _find_total_points_tag(ea.Tags().get("scalars", []))
        if tag is None:
            print(f"WARN: no total_points tag in {sd}")
            continue
        for step in args.step:
            value = _read_total_points_at_step(events_dir, tag, step)
            if value is not None:
                table.append({"seed": _seed_name(sd), "step": step, "value": int(value)})

    if not table:
        print("ERROR: no total_points data found")
        sys.exit(1)

    # Print table
    header = f"{'seed':>12}  {'step':>8}  {'total_points':>14}"
    print(header)
    print("-" * len(header))
    for row in table:
        print(f"{row['seed']:>12}  {row['step']:>8}  {row['value']:>14}")

    # Aggregate
    if args.aggregate:
        by_step = {}
        for row in table:
            by_step.setdefault(row["step"], []).append(row["value"])
        print()
        for step, vals in sorted(by_step.items()):
            mu = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"step {step}: mean={mu:,.0f}  stdev={stdev:,.0f}  n={len(vals)}")
            # Write BTOTAL file
            if args.output_btotal:
                # Only write the last step (or the one requested)
                requested = set(args.step)
                if step in requested and len(requested) == 1:
                    with open(args.output_btotal, 'w') as f:
                        f.write(str(int(mu)))
                    print(f"  → wrote {args.output_btotal}")


# ---------------------------------------------------------------------------
# Subcommand: metrics
# ---------------------------------------------------------------------------

def _find_checkpoints_in_results(results):
    """Return sorted checkpoint keys like ['ours_7000','ours_15000',...] from results dict."""
    cps = []
    for k in results:
        if k.startswith("ours_"):
            try:
                cps.append((int(k.split("_")[1]), k))
            except (ValueError, IndexError):
                pass
    cps.sort()
    return [k for _, k in cps]


def cmd_metrics(args):
    seeds = _seed_dirs(args.glob)
    if not seeds:
        print(f"ERROR: no seed dirs match {args.glob}")
        sys.exit(1)

    per_seed = {}    # seed → {cp → {PSNR,SSIM,LPIPS}}
    per_cp_raw = {}  # cp → {PSNR: [vals], SSIM: [...], LPIPS: [...]}

    for sd in seeds:
        results_path = os.path.join(sd, "results.json")
        if not os.path.exists(results_path):
            print(f"WARN: no results.json in {sd}")
            continue
        data = _read_results_json(results_path)
        seed = _seed_name(sd)
        per_seed[seed] = {}

        # Filter checkpoints
        cps = _find_checkpoints_in_results(data)
        if args.checkpoints:
            cps = [cp for cp in cps if any(
                str(chk) in cp for chk in args.checkpoints)]
        if not cps and args.checkpoints:
            # Try exact match
            cps = [f"ours_{cp}" for cp in args.checkpoints if f"ours_{cp}" in data]

        for cp in cps:
            if cp not in data:
                continue
            m = data[cp]
            per_seed[seed][cp] = {"PSNR": m["PSNR"], "SSIM": m["SSIM"], "LPIPS": m["LPIPS"]}
            per_cp_raw.setdefault(cp, {"PSNR": [], "SSIM": [], "LPIPS": []})
            for metric in ("PSNR", "SSIM", "LPIPS"):
                per_cp_raw[cp][metric].append(m[metric])

    # Per-seed table
    print(f"{'seed':>12}  {'checkpoint':>12}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
    print("-" * 64)
    for seed in sorted(per_seed):
        for cp in sorted(per_seed[seed]):
            m = per_seed[seed][cp]
            print(f"{seed:>12}  {cp:>12}  {m['PSNR']:>8.4f}  {m['SSIM']:>8.4f}  {m['LPIPS']:>8.4f}")

    # Aggregate summary
    print()
    summary = {}
    for cp in sorted(per_cp_raw):
        raw = per_cp_raw[cp]
        entry = {}
        for metric in ("PSNR", "SSIM", "LPIPS"):
            vals = raw[metric]
            mu = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            entry[metric] = {"mean": mu, "stdev": stdev, "n": len(vals)}
        summary[cp] = entry
        print(f"{cp}: PSNR={entry['PSNR']['mean']:.4f}±{entry['PSNR']['stdev']:.4f}  "
              f"SSIM={entry['SSIM']['mean']:.4f}±{entry['SSIM']['stdev']:.4f}  "
              f"LPIPS={entry['LPIPS']['mean']:.4f}±{entry['LPIPS']['stdev']:.4f}  "
              f"n={entry['PSNR']['n']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({"seeds": {s: per_seed[s] for s in sorted(per_seed)},
                       "summary": summary}, f, indent=2)
        print(f"\nWrote {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def _load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)


def cmd_compare(args):
    a_data = _load_summary(args.a_only)
    b_data = _load_summary(args.a_plus_b)
    sigma_data = _load_summary(args.sigma)

    a_summary = a_data.get("summary", a_data)
    b_summary = b_data.get("summary", b_data)
    sigma_summary = sigma_data.get("summary", sigma_data)

    checkpoints = sorted(set(a_summary.keys()) & set(b_summary.keys()))

    decision = "DROP"
    best_delta_sigma = 0.0

    print(f"{'checkpoint':>12}  {'μ_A':>8}  {'μ_B':>8}  {'Δ':>8}  {'2σ':>8}  {'|Δ|>2σ?':>10}")
    print("-" * 64)

    for cp in checkpoints:
        if cp not in sigma_summary:
            continue
        mu_a = a_summary[cp]["PSNR"]["mean"]
        mu_b = b_summary[cp]["PSNR"]["mean"]
        sigma_psnr = sigma_summary[cp]["PSNR"]["stdev"]
        delta = mu_b - mu_a
        two_sigma = 2.0 * sigma_psnr
        significant = abs(delta) > two_sigma

        print(f"{cp:>12}  {mu_a:>8.4f}  {mu_b:>8.4f}  {delta:>+8.4f}  {two_sigma:>8.4f}  "
              f"{'YES' if significant else 'no':>10}")

        if significant:
            decision = "KEEP"
        ratio = abs(delta) / sigma_psnr if sigma_psnr > 0 else 0.0
        best_delta_sigma = max(best_delta_sigma, ratio)

    print()
    print(f"Max |Δ|/σ = {best_delta_sigma:.2f}")
    print(f"Decision: {decision} B")
    if decision == "DROP":
        print("  B signal indistinguishable from noise → fallback per ADR-0002")
    else:
        print("  B signal detectable above noise floor → KEEP, proceed to fidelity sweep (Step 5)")


# ---------------------------------------------------------------------------
# Subcommand: table
# ---------------------------------------------------------------------------

def _parse_table_args(argv):
    parser = argparse.ArgumentParser(prog="table")
    parser.add_argument("--arm", type=str, action="append", required=True,
                        help="Arm label=path pairs (repeatable)")
    parser.add_argument("--baseline-label", default=None,
                        help="Label of the arm to use as Δ baseline")
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "LPIPS"],
                        help="Metrics to tabulate (default PSNR SSIM LPIPS)")
    if argv and argv[0] == "table":
        argv = argv[1:]
    return parser.parse_args(argv)


def cmd_table(args):
    arms = {}
    for entry in args.arm:
        if "=" not in entry:
            print(f"ERROR: --arm must be label=path, got '{entry}'")
            sys.exit(1)
        label, path = entry.split("=", 1)
        if label in arms:
            print(f"ERROR: duplicate arm label '{label}'")
            sys.exit(1)
        arms[label] = path

    if len(arms) < 1:
        print("ERROR: at least one --arm required")
        sys.exit(1)

    data = {}
    for label, path in arms.items():
        if not os.path.exists(path):
            print(f"ERROR: arm '{label}' file not found: {path}")
            sys.exit(1)
        d = _load_summary(path)
        data[label] = d.get("summary", d)

    if args.baseline_label and args.baseline_label not in data:
        print(f"ERROR: baseline label '{args.baseline_label}' not found in arms")
        sys.exit(1)

    checkpoints = set()
    for label in data:
        checkpoints.update(data[label].keys())
    if not checkpoints:
        print("(no checkpoints found)")
        return
    checkpoints = sorted(checkpoints)

    baseline_label = args.baseline_label if args.baseline_label else list(arms.keys())[0]
    labels = list(arms.keys())
    metrics = args.metrics

    for metric in metrics:
        print(f"\n=== {metric} ===")

        header_cols = ["checkpoint"]
        for label in labels:
            header_cols.append(f"{label}_μ")
            if label != baseline_label:
                header_cols.append(f"Δ_{label}")

        header = "  ".join(f"{c:>14}" for c in header_cols)
        print(header)
        print("-" * len(header))

        for cp in checkpoints:
            ref_entry = data[baseline_label].get(cp)
            ref_val = ref_entry[metric]["mean"] if ref_entry else None

            row = [f"{cp:>14}"]
            for label in labels:
                entry = data[label].get(cp)
                if entry is None:
                    row.append(f"{'—':>14}")
                    if label != baseline_label:
                        row.append(f"{'—':>14}")
                    continue
                val = entry[metric]["mean"]
                row.append(f"{val:>14.4f}")
                if label != baseline_label:
                    if ref_val is not None:
                        delta = val - ref_val
                        row.append(f"{delta:>+14.4f}")
                    else:
                        row.append(f"{'—':>14}")

            print("  ".join(row))

    print()


# ---------------------------------------------------------------------------
# Subcommand: pareto
# ---------------------------------------------------------------------------

def aggregate_pareto_points(raw):
    """Collapse per-(arm,factor) seed lists into one Pareto row each.

    raw: list of {arm, factor, anchors:[int], metrics:[{PSNR,SSIM,LPIPS}]}.
    Returns rows {arm, factor, anchors_mean, PSNR_mean, SSIM_mean, LPIPS_mean, n}
    sorted by (arm, anchors_mean). Points with no anchors or no metrics are
    dropped (an infeasible / missing factor).
    """
    rows = []
    for pt in raw:
        anchors = pt.get("anchors") or []
        metrics = pt.get("metrics") or []
        if not anchors or not metrics:
            continue
        row = {
            "arm": pt["arm"],
            "factor": pt["factor"],
            "anchors_mean": statistics.mean(anchors),
            "n": len(metrics),
        }
        for m in ("PSNR", "SSIM", "LPIPS"):
            row[f"{m}_mean"] = statistics.mean(d[m] for d in metrics)
        rows.append(row)
    rows.sort(key=lambda r: (r["arm"], r["anchors_mean"]))
    return rows


def cmd_pareto(args):
    raw = []
    for arm in args.arms:
        for factor in args.factors:
            label = f"{factor:g}x"
            armdir = os.path.join(args.root, arm, f"arm_{label}")
            seeds = _seed_dirs(os.path.join(armdir, "seed_*"))
            anchors, metrics = [], []
            for sd in seeds:
                events_dir = _find_events_dir(sd)
                if events_dir is not None:
                    ea = EventAccumulator(events_dir)
                    ea.Reload()
                    tag = _find_total_points_tag(ea.Tags().get("scalars", []))
                    if tag is not None:
                        v = _read_total_points_at_step(events_dir, tag, args.step)
                        if v is not None:
                            anchors.append(int(v))
                if not anchors:
                    log_path = os.path.join(sd, "outputs.log")
                    if os.path.exists(log_path):
                        with open(log_path) as lf:
                            for line in lf:
                                if "final anchors=" in line:
                                    try:
                                        anchors.append(int(line.split("final anchors=")[1].split()[0]))
                                    except (ValueError, IndexError):
                                        pass
                                    break
                rp = os.path.join(sd, "results.json")
                if os.path.exists(rp):
                    data = _read_results_json(rp)
                    key = f"ours_{args.checkpoint}"
                    if key in data:
                        m = data[key]
                        metrics.append({"PSNR": m["PSNR"], "SSIM": m["SSIM"],
                                        "LPIPS": m["LPIPS"]})
            raw.append({"arm": arm, "factor": factor,
                        "anchors": anchors, "metrics": metrics})

    rows = aggregate_pareto_points(raw)
    header = "arm,factor,anchors,PSNR,SSIM,LPIPS,n"
    lines = [header]
    print(header)
    for r in rows:
        line = (f"{r['arm']},{r['factor']:g},{r['anchors_mean']:.0f},"
                f"{r['PSNR_mean']:.4f},{r['SSIM_mean']:.4f},"
                f"{r['LPIPS_mean']:.4f},{r['n']}")
        lines.append(line)
        print(line)
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 4 result collector")
    sub = parser.add_subparsers(dest="command")

    # total_points
    p_tp = sub.add_parser("total_points", help="Extract total_points from TB events")
    p_tp.add_argument("--glob", required=True, help="Glob pattern for seed dirs")
    p_tp.add_argument("--step", type=int, nargs="+", required=True, help="Iteration step(s)")
    p_tp.add_argument("--tag-suffix", default="total_points", help="Suffix of TB scalar tag")
    p_tp.add_argument("--aggregate", choices=["mean"], default=None)
    p_tp.add_argument("--output-btotal", default=None, help="Write mean B_total to file")

    # metrics
    p_m = sub.add_parser("metrics", help="Collect PSNR/SSIM/LPIPS from results.json")
    p_m.add_argument("--glob", required=True, help="Glob pattern for seed dirs")
    p_m.add_argument("--checkpoints", type=int, nargs="+", help="Iterations (e.g. 7000 15000 25000 30000)")
    p_m.add_argument("--output", default=None, help="Write summary JSON")

    # compare
    p_c = sub.add_parser("compare", help="Compare A-only vs A+B vs sigma")
    p_c.add_argument("--a-only", required=True, help="A-only summary JSON")
    p_c.add_argument("--a-plus-b", required=True, help="A+B summary JSON")
    p_c.add_argument("--sigma", required=True, help="Sigma summary JSON (A-only noise floor)")

    # table
    p_t = sub.add_parser("table", help="Tabulate metrics across multiple arms")
    p_t.add_argument("--arm", type=str, action="append", required=True,
                     help="Arm label=path pairs (repeatable, e.g. --arm baseline=summary.json)")
    p_t.add_argument("--baseline-label", default=None,
                     help="Label of the arm to use as Δ baseline")
    p_t.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "LPIPS"],
                     help="Metrics to tabulate (default: PSNR SSIM LPIPS)")

    # pareto
    p_p = sub.add_parser("pareto", help="Join achieved anchors (x) with quality (y) per arm/factor")
    p_p.add_argument("--root", required=True, help="Sweep root: <root>/<arm>/arm_<factor>x/seed_*")
    p_p.add_argument("--arms", nargs="+", required=True, help="Arm names (e.g. demand uniform)")
    p_p.add_argument("--factors", type=float, nargs="+", required=True, help="Budget factors")
    p_p.add_argument("--step", type=int, required=True, help="total_points step (update_until)")
    p_p.add_argument("--checkpoint", type=int, required=True, help="metrics checkpoint (e.g. 30000)")
    p_p.add_argument("--output", default=None, help="Write CSV")

    args = parser.parse_args()
    if args.command == "total_points":
        cmd_total_points(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "table":
        cmd_table(args)
    elif args.command == "pareto":
        cmd_pareto(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
