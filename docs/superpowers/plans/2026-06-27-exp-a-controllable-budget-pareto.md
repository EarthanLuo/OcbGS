# Exp A — Controllable-Budget Quality–Cost Pareto Front Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the "money figure" of the mainline — a training-side quality–cost Pareto front on BungeeNeRF/amsterdam, comparing a gradient-**demand** allocation curve against a **uniform** allocation curve, both swept over `B_total ∈ {0.25, 0.5, 1, 2}×` baseline under identical controller machinery.

**Architecture:** The controller already supports demand allocation and (via its `d_sum<=0 → uniform` branch) uniform allocation; the `--grow_relax_scale 0.1` actuator switch lets it actually hit its budget set-point (amsterdam fills 99.7% of `B_total`). This plan adds a `--demand_uniform` flag to flatten demand into the controller's uniform branch, an amsterdam sweep driver adapted from `scripts/exp2_garden_pareto.sh`, a `pareto` collator and a plot script, and a held-out-test confirmation pass — then reconciles `eval-plan.md` with the retired matched-budget arm.

**Tech Stack:** Python 3.12 / PyTorch 2.5.1 (server, RTX 4090, sm_89); pytest for pure-logic local tests; bash sweep drivers; matplotlib for the Pareto plot; TensorBoard event files + `results.json` as the data source.

## Global Constraints

- **Demand arm is A-only.** Source B is a tested negative result (far-view ΔPSNR −0.017 train / −0.218 held-out test); it ships demoted. The main sweep's demand arm uses gradient demand only (`b_enabled=False`, the default). Do **not** add B to the Pareto sweep.
- **No-force-fill invariant (ADR-0003/0004/0005) holds.** `δ⁺` is an upper bound; executed occupancy is `≤ B_total` by design. The Pareto curve is plotted on **achieved** budget. Never reintroduce a "floor fills the budget" / matched-budget force-fill path.
- **`--grow_relax_scale 0.1`** on every controller arm of the sweep (demand and uniform), so both hit the set-point. `1.0` = native gating (undershoot). Locked at `0.1` for amsterdam (already validated: 99.7% fill).
- **Eval protocol:** the swept curve uses **train-set** metrics (no `--eval`); a **held-out test** confirmation (`--eval`) is rendered from the saved checkpoints only at the locked operating point(s). Test appearance uses shifted train-uid embeddings (`--ape -1`) — absolute test PSNR is approximate but the demand-vs-uniform comparison is fair. "Proper test-time appearance handling" is owed for the final paper (out of scope here).
- **`B_total` = baseline anchor count at `update_until` (25000)**, measured once per scene with `--no_controller` and a fixed seed (eval-plan §4). Never compare a controller run stopped at a different horizon to it.
- **Scope: amsterdam first** (`/root/autodl-tmp/bungeenerf/amsterdam`; `BTOTAL` file at `/root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam`). Extend to quebec/rome only after the amsterdam double-curve is clean.
- **Server pitfalls (CLAUDE.md):** single GPU → `MAX_JOBS=1`; `set -e` + bare `(( var++ ))` aborts — always `(( _running++ )) || true`; per-seed resume guard on every arm; `--ds N` (not `-i images_N`).
- **Local vs server:** all pure-logic tests (helpers, collator, plot data-shaping) run locally with CPU tensors / synthetic data — no CUDA. All training/rendering/smoke runs are handed to the user with exact commands; never mark a GPU step done without the user's pasted, completed output (superpowers:verification-before-completion).
- **Language:** every file written is in English; chat is Chinese.

---

### Task 1: `--demand_uniform` flag and uniform-demand hook

Adds the mechanism that produces the **uniform** Pareto arm under identical controller machinery: a flag that zeros the per-cell demand `d_a` before it reaches the controller, so the existing `d_sum<=0 → B_total/N_active` branch (`controller/__init__.py:86-87`) fires. Pure-logic helper is unit-tested locally; the GPU integration smoke is handed to the user.

**Files:**
- Modify: `ocbgs/controller/__init__.py` (add `resolve_controller_demand` helper near `align_demand_b`, ~line 404)
- Modify: `ocbgs/arguments/__init__.py:177` (add `--demand_uniform` next to `grow_relax_scale`)
- Modify: `ocbgs/scene/gaussian_model.py:449` (read the flag) and `:1313` (apply the helper at the `controller.plan` call)
- Test: `ocbgs/tests/test_demand_uniform.py`

**Interfaces:**
- Produces: `resolve_controller_demand(d_a: torch.Tensor, uniform: bool) -> torch.Tensor` — returns `torch.zeros_like(d_a)` when `uniform` is True, else `d_a` unchanged. Consumed by `GaussianModel.adjust_anchor`.
- Produces: training arg `demand_uniform: bool` (default `False`), read as `self.demand_uniform` on the model.

- [ ] **Step 1: Write the failing test**

Create `ocbgs/tests/test_demand_uniform.py`:

```python
import torch
from controller import resolve_controller_demand, StaticBudgetController


def test_resolve_returns_demand_unchanged_when_not_uniform():
    d_a = torch.tensor([3.0, 1.0, 0.0, 5.0])
    out = resolve_controller_demand(d_a, uniform=False)
    assert torch.equal(out, d_a)


def test_resolve_zeros_demand_when_uniform():
    d_a = torch.tensor([3.0, 1.0, 0.0, 5.0])
    out = resolve_controller_demand(d_a, uniform=True)
    assert torch.equal(out, torch.zeros_like(d_a))
    assert out.shape == d_a.shape


def test_zeroed_demand_drives_controller_uniform_branch():
    # With all-zero demand, _allocate must spread B_total evenly over active cells
    # (the d_sum<=0 -> B_total/N_active branch), independent of demand shape.
    ctrl = StaticBudgetController(floor=1, k_cap=8)
    cell_ids = torch.arange(4)
    occupancy = torch.tensor([10, 10, 10, 10])
    d_zero = resolve_controller_demand(torch.tensor([9.0, 1.0, 1.0, 1.0]), uniform=True)
    plan = ctrl.plan(cell_ids, d_A=d_zero, occupancy=occupancy, B_total=40)
    # 40 budget / 4 active cells = 10 each -> c_target is flat
    assert plan.c_target.tolist() == [10, 10, 10, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ocbgs/tests/test_demand_uniform.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_controller_demand' from 'controller'`.

- [ ] **Step 3: Add the helper**

In `ocbgs/controller/__init__.py`, immediately above `def align_demand_b` (~line 404), add:

```python
def resolve_controller_demand(d_a, uniform):
    """Return the demand field fed to the controller.

    When ``uniform`` is True, zero the demand so the controller's
    ``d_sum<=0 -> B_total/N_active`` branch produces a flat allocation across
    active cells — the 'uniform' Pareto arm, run under identical machinery as
    the demand arm. When False, the gradient demand passes through unchanged.
    """
    if uniform:
        return torch.zeros_like(d_a)
    return d_a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ocbgs/tests/test_demand_uniform.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Wire the argument**

In `ocbgs/arguments/__init__.py`, after line 177 (`self.grow_relax_scale = 1.0`), add:

```python
        self.demand_uniform = False
```

(It is an `ArgumentGroup` member, so argparse exposes it automatically as `--demand_uniform`; the default `False` makes it a store-true-style flag via the codebase's `ParamGroup` boolean handling.)

- [ ] **Step 6: Read the flag on the model**

In `ocbgs/scene/gaussian_model.py`, after line 449 (`self.grow_relax_scale = getattr(...)`), add:

```python
        # Uniform-allocation control arm: zero the demand field so the
        # controller falls back to B_total/N_active (Exp A uniform curve).
        self.demand_uniform = getattr(training_args, 'demand_uniform', False)
```

- [ ] **Step 7: Apply the helper at the controller call**

In `ocbgs/scene/gaussian_model.py`, change the `controller.plan` call (line 1313) from:

```python
        plan = self.controller.plan(cell_ids=cids, d_A=d_a, occupancy=n,
                                     B_total=self.B_total, d_B=d_b)
```

to:

```python
        d_a_eff = resolve_controller_demand(d_a, self.demand_uniform)
        plan = self.controller.plan(cell_ids=cids, d_A=d_a_eff, occupancy=n,
                                     B_total=self.B_total, d_B=d_b)
```

and add `resolve_controller_demand` to the existing controller import at line 33:

```python
from controller import build_controller, align_demand_b, resolve_controller_demand
```

- [ ] **Step 8: Re-run the local test**

Run: `python -m pytest ocbgs/tests/test_demand_uniform.py -v`
Expected: PASS (3 passed).

- [ ] **Step 9: Commit**

```bash
git add ocbgs/controller/__init__.py ocbgs/arguments/__init__.py ocbgs/scene/gaussian_model.py ocbgs/tests/test_demand_uniform.py
git commit -m "feat: --demand_uniform flag flattens demand into controller uniform branch (Exp A uniform arm)"
```

- [ ] **Step 10: Hand the GPU smoke to the user (review checkpoint)**

Hand off — the user runs a short smoke on the server to confirm the flag flattens allocation **and** still hits the budget under relax (CLAUDE.md fast-inner-loop rule: this answers "did the mechanism move?", not quality, so a smoke suffices):

```bash
# amsterdam, uniform arm, compressed controller window, relax on
python ocbgs/train.py \
    -s /root/autodl-tmp/bungeenerf/amsterdam --ds 1 \
    -m /root/autodl-tmp/expA/smoke/uniform_relax01 \
    --fork 2 --base_layer 10 --visible_threshold 0.0 \
    --dist2level round --update_ratio 0.2 \
    --iterations 3000 --update_from 500 --update_interval 100 --update_until 3000 \
    --seed 0 --port 6021 \
    --B_total $(cat /root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam) \
    --grow_relax_scale 0.1 --demand_uniform

python scripts/collect_results.py total_points \
    --glob /root/autodl-tmp/expA/smoke/uniform_relax01 --step 3000 --aggregate mean
```

Expected (paste back): the run completes; the printed `total_points@3000` is within a few % of (a compressed-horizon fraction of) `B_total` and the per-cell target is visibly flat (uniform). Do **not** mark this step done without the pasted output. If the uniform arm undershoots far more than the demand arm at the same relax, stop and re-localise with data (do not guess the next knob).

---

### Task 2: amsterdam Pareto sweep driver

Adapts `scripts/exp2_garden_pareto.sh` into a bungeenerf/amsterdam driver that sweeps **two arms** (demand A-only, uniform) across `{0.25,0.5,1,2}×`, with `--grow_relax_scale 0.1` on every controller run, single-GPU concurrency guards, and per-seed resume. Bash — handed to the user to run on the server; locally we only self-review it against the CLAUDE.md pitfalls.

**Files:**
- Create: `scripts/exp_a_amsterdam_pareto.sh`

**Interfaces:**
- Consumes: `BTOTAL_amsterdam` (Task's Phase 0, or the pre-measured file); `--demand_uniform` (Task 1); `--grow_relax_scale` (existing).
- Produces: run dirs `/root/autodl-tmp/expA/amsterdam/<arm>/arm_<factor>x/seed_*` for `arm ∈ {demand, uniform}`, each with `results.json` (train-set metrics) and TB events (for `total_points`). Consumed by Task 4.

- [ ] **Step 1: Write the sweep script**

Create `scripts/exp_a_amsterdam_pareto.sh`:

```bash
#!/bin/bash
# Exp A — amsterdam controllable-budget Pareto: demand vs uniform.
#
# Two arms, both with --grow_relax_scale 0.1 so the controller hits B_total:
#   demand   : gradient demand allocation (A-only; b_enabled default off)
#   uniform  : --demand_uniform -> controller B_total/N_active flat allocation
# Sweep B_total x {0.25,0.5,1,2}x baseline. Train-set metrics (no --eval);
# held-out test is rendered separately at the locked point (Task 6).
#
# set -e + bare (( var++ )) aborts on the first 0->1 increment, so every
# counter mutation is guarded with || true (CLAUDE.md). Per-seed resume guard
# skips any seed whose results.json already exists.

set -e
export PYTHONWARNINGS=ignore

SRC=/root/autodl-tmp/bungeenerf/amsterdam
DST=/root/autodl-tmp/expA/amsterdam
BTOTAL_FILE=/root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam
ITERS=30000
UPDATE_UNTIL=25000
CHECKPOINTS=(30000)
SAVE_CHECKPOINTS=(25000 30000)
SEEDS=(0 1 2)
MAX_JOBS=${MAX_JOBS:-1}          # single GPU
FACTORS=(0.25 0.5 1 2)
RELAX=0.1

mkdir -p "$DST"

if [ ! -f "$BTOTAL_FILE" ]; then
    echo "ERROR: $BTOTAL_FILE missing — measure B_total first (eval-plan §4)."
    exit 1
fi
B_TOTAL=$(cat "$BTOTAL_FILE")
echo "=== Exp A amsterdam Pareto ===  B_total=$B_TOTAL  relax=$RELAX"

run_arm () {
    # $1 = arm name (demand|uniform); $2... = extra train.py flags
    local arm="$1"; shift
    local extra=("$@")
    for factor in "${FACTORS[@]}"; do
        local B_val
        B_val=$(python -c "print(int($B_TOTAL * $factor))")
        local armdir="$DST/$arm/arm_${factor}x"
        mkdir -p "$armdir"
        echo ""
        echo "--- arm=$arm factor=$factor (B_total=$B_val) ---"

        # Feasibility probe on seed_0 (synchronous): set_control_level's guard
        # raises a non-zero exit if 0.25x is infeasible for this scene.
        if [ -f "$armdir/seed_0/results.json" ]; then
            echo "  seed=0 — DONE (skip probe)"
        else
            echo "  seed=0 (feasibility probe)"
            if ! python ocbgs/train.py \
                -s "$SRC" --ds 1 -m "$armdir/seed_0" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed 0 --port $((6009 + RANDOM % 1000)) \
                --B_total $B_val --grow_relax_scale $RELAX "${extra[@]}"; then
                echo "  arm=$arm factor=$factor INFEASIBLE — skip"
                continue
            fi
        fi

        local _running=0
        for seed in "${SEEDS[@]}"; do
            [ "$seed" -eq 0 ] && continue
            if [ -f "$armdir/seed_$seed/results.json" ]; then
                echo "  seed=$seed — DONE (skip)"; continue
            fi
            while (( _running >= MAX_JOBS )); do
                wait -n 2>/dev/null || true
                (( _running-- )) || true
            done
            echo "  seed=$seed"
            python ocbgs/train.py \
                -s "$SRC" --ds 1 -m "$armdir/seed_$seed" \
                --fork 2 --base_layer 10 --visible_threshold 0.0 \
                --dist2level round --update_ratio 0.2 \
                --iterations $ITERS --update_until $UPDATE_UNTIL \
                --test_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --save_iterations "${SAVE_CHECKPOINTS[@]}" $ITERS \
                --seed $seed --port $((6009 + RANDOM % 1000)) \
                --B_total $B_val --grow_relax_scale $RELAX "${extra[@]}" &
            (( _running++ )) || true
            sleep 10s
        done
        wait
    done
}

run_arm demand
run_arm uniform --demand_uniform

echo ""
echo "=== Exp A amsterdam sweep done — collate with: ==="
echo "python scripts/collect_results.py pareto --root $DST \\"
echo "    --arms demand uniform --factors ${FACTORS[*]} \\"
echo "    --step $UPDATE_UNTIL --checkpoint ${CHECKPOINTS[0]} --output $DST/pareto.csv"
```

- [ ] **Step 2: Self-review the script against the pitfall checklist**

Confirm by reading the script: (a) every `(( _running++ ))` / `(( _running-- ))` has `|| true`; (b) every arm/seed has a `results.json` resume guard; (c) `MAX_JOBS` defaults to 1; (d) `--ds 1` (not `-i`); (e) `--grow_relax_scale 0.1` is on every `train.py` invocation; (f) no `--eval` (train-set curve); (g) the uniform arm passes `--demand_uniform` and the demand arm does not. Fix any miss inline.

- [ ] **Step 3: Commit**

```bash
git add scripts/exp_a_amsterdam_pareto.sh
git commit -m "feat: Exp A amsterdam Pareto sweep driver (demand vs uniform, relax 0.1)"
```

- [ ] **Step 4: Hand the sweep to the user (review checkpoint — long run)**

This is the expensive outer loop (2 arms × 4 factors × 3 seeds = up to 24 full 30k trainings, single GPU, run sequentially). Do **not** launch it until Task 1's smoke (Step 10) and Task 3's relax-fill confirmation pass. Hand off:

```bash
chmod +x scripts/exp_a_amsterdam_pareto.sh
MAX_JOBS=1 bash scripts/exp_a_amsterdam_pareto.sh 2>&1 | tee /root/autodl-tmp/expA/amsterdam/sweep.log
```

Expected (paste back): both arms complete across all feasible factors; each `seed_*/results.json` exists; `sweep.log` ends at "Exp A amsterdam sweep done". Any `INFEASIBLE` factor is reported and skipped (expected possible at 0.25×). Mark done only on the pasted completion.

---

### Task 3: Relax-fill confirmation (gating smoke)

Before committing to the 24-run sweep, confirm at smoke length that **both** arms hit ~`B_total` at `relax=0.1` and at the tightest/loosest factors — so the curve points land at the intended budgets, not at an undershoot. This is a cheap gate per the experiment-workflow rule; it answers "did the budget fill?", which is horizon-independent.

**Files:** none (uses Task 1/2 code).

- [ ] **Step 1: Hand the smoke matrix to the user**

For `arm ∈ {demand, uniform}` × `factor ∈ {0.25, 2}` (the extremes), run the compressed-window smoke and read the fill ratio:

```bash
B=$(cat /root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam)
for arm_flag in "demand:" "uniform:--demand_uniform"; do
  arm=${arm_flag%%:*}; flag=${arm_flag#*:}
  for f in 0.25 2; do
    Bv=$(python -c "print(int($B * $f))")
    out=/root/autodl-tmp/expA/fillsmoke/${arm}_${f}x
    python ocbgs/train.py -s /root/autodl-tmp/bungeenerf/amsterdam --ds 1 \
      -m "$out" --fork 2 --base_layer 10 --visible_threshold 0.0 \
      --dist2level round --update_ratio 0.2 \
      --iterations 3000 --update_from 500 --update_interval 100 --update_until 3000 \
      --seed 0 --port $((6009 + RANDOM % 1000)) \
      --B_total $Bv --grow_relax_scale 0.1 $flag
    echo "== $arm $f x (B_total=$Bv) =="
    python scripts/collect_results.py total_points --glob "$out" --step 3000 --aggregate mean
  done
done
```

- [ ] **Step 2: Review the fill ratios (gate decision)**

Compute `total_points@3000 / B_val` for each of the 4 cells. Expected: all near the controllable ceiling (amsterdam at `relax=0.1` reached 99.7% at 1× full-length; smoke at compressed horizon will be lower in absolute fill but should be **comparable between demand and uniform** at each factor). Go criterion: demand and uniform fill within a few % of each other at each factor (a fair x-axis comparison needs both arms reaching similar achieved budgets). If one arm systematically undershoots, **stop and re-localise with data** (e.g. inspect the controller phase logs) before running the full sweep — do not bump relax blindly.

---

### Task 4: `pareto` collator subcommand

Joins each (arm, factor) point's **achieved anchors** (x, from TB `total_points@update_until`) with its **quality** (y, mean±std PSNR/SSIM/LPIPS from `results.json` at the checkpoint) into a tidy CSV the plot script consumes. The aggregation core is pure-logic and unit-tested; the CLI wrapper reuses existing disk helpers.

**Files:**
- Modify: `scripts/collect_results.py` (add `aggregate_pareto_points` + `cmd_pareto` + the `pareto` subparser)
- Test: `scripts/test_collect_pareto.py`

**Interfaces:**
- Produces: `aggregate_pareto_points(raw: list[dict]) -> list[dict]` where each input dict is `{"arm": str, "factor": float, "anchors": list[int], "metrics": list[dict]}` (each metrics dict has `PSNR`/`SSIM`/`LPIPS`) and each output row is `{"arm", "factor", "anchors_mean", "PSNR_mean", "SSIM_mean", "LPIPS_mean", "n"}`, sorted by `(arm, anchors_mean)`.
- Produces: CLI `pareto --root <dir> --arms <a..> --factors <f..> --step <int> --checkpoint <int> --output <csv>` writing `arm,factor,anchors,PSNR,SSIM,LPIPS,n` rows. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_collect_pareto.py`:

```python
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "collect_results",
    os.path.join(os.path.dirname(__file__), "collect_results.py"),
)
collect_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_results)
aggregate_pareto_points = collect_results.aggregate_pareto_points


def test_aggregates_mean_and_sorts_by_arm_then_anchors():
    raw = [
        {"arm": "uniform", "factor": 1.0, "anchors": [100, 120],
         "metrics": [{"PSNR": 28.0, "SSIM": 0.80, "LPIPS": 0.20},
                     {"PSNR": 28.4, "SSIM": 0.82, "LPIPS": 0.18}]},
        {"arm": "demand", "factor": 0.5, "anchors": [50, 50],
         "metrics": [{"PSNR": 27.0, "SSIM": 0.78, "LPIPS": 0.22},
                     {"PSNR": 27.0, "SSIM": 0.78, "LPIPS": 0.22}]},
        {"arm": "demand", "factor": 1.0, "anchors": [110, 90],
         "metrics": [{"PSNR": 28.5, "SSIM": 0.83, "LPIPS": 0.17},
                     {"PSNR": 28.5, "SSIM": 0.83, "LPIPS": 0.17}]},
    ]
    rows = aggregate_pareto_points(raw)
    # sorted by (arm, anchors_mean): demand@50, demand@100, uniform@110
    assert [(r["arm"], r["anchors_mean"]) for r in rows] == [
        ("demand", 50.0), ("demand", 100.0), ("uniform", 110.0)]
    assert rows[0]["PSNR_mean"] == 27.0
    assert rows[0]["n"] == 2
    assert abs(rows[2]["PSNR_mean"] - 28.2) < 1e-9


def test_skips_points_with_no_data():
    raw = [{"arm": "demand", "factor": 0.25, "anchors": [], "metrics": []}]
    assert aggregate_pareto_points(raw) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/test_collect_pareto.py -v`
Expected: FAIL with `AttributeError: module 'collect_results' has no attribute 'aggregate_pareto_points'`.

- [ ] **Step 3: Add the aggregation helper**

In `scripts/collect_results.py`, after `cmd_table` (before the `# CLI` section, ~line 384), add:

```python
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
```

- [ ] **Step 4: Register the subparser**

In `scripts/collect_results.py`, inside `main()` after the `table` subparser block (~line 421), add:

```python
    # pareto
    p_p = sub.add_parser("pareto", help="Join achieved anchors (x) with quality (y) per arm/factor")
    p_p.add_argument("--root", required=True, help="Sweep root: <root>/<arm>/arm_<factor>x/seed_*")
    p_p.add_argument("--arms", nargs="+", required=True, help="Arm names (e.g. demand uniform)")
    p_p.add_argument("--factors", type=float, nargs="+", required=True, help="Budget factors")
    p_p.add_argument("--step", type=int, required=True, help="total_points step (update_until)")
    p_p.add_argument("--checkpoint", type=int, required=True, help="metrics checkpoint (e.g. 30000)")
    p_p.add_argument("--output", default=None, help="Write CSV")
```

and in the dispatch chain at the bottom of `main()`, after the `table` branch:

```python
    elif args.command == "pareto":
        cmd_pareto(args)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest scripts/test_collect_pareto.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_results.py scripts/test_collect_pareto.py
git commit -m "feat: collect_results.py pareto — join achieved anchors with quality per arm/factor"
```

---

### Task 5: Pareto plot script

Reads the `pareto.csv` from Task 4 and draws the two curves (demand vs uniform: x = achieved anchors, y = PSNR), the money figure. The curve-splitting logic is pure-logic and unit-tested; matplotlib drawing is a thin wrapper.

**Files:**
- Create: `scripts/plot_pareto.py`
- Test: `scripts/test_plot_pareto.py`

**Interfaces:**
- Consumes: the CSV produced by Task 4 (`arm,factor,anchors,PSNR,SSIM,LPIPS,n`).
- Produces: `load_pareto_rows(path: str) -> list[dict]` and `split_curves(rows: list[dict], metric: str) -> dict[str, tuple[list, list]]` mapping arm → (xs sorted by anchors, ys).

- [ ] **Step 1: Write the failing test**

Create `scripts/test_plot_pareto.py`:

```python
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "plot_pareto",
    os.path.join(os.path.dirname(__file__), "plot_pareto.py"),
)
plot_pareto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_pareto)


def test_split_curves_groups_by_arm_sorted_by_anchors():
    rows = [
        {"arm": "demand", "anchors": 100.0, "PSNR": 28.5},
        {"arm": "uniform", "anchors": 110.0, "PSNR": 28.2},
        {"arm": "demand", "anchors": 50.0, "PSNR": 27.0},
    ]
    curves = plot_pareto.split_curves(rows, metric="PSNR")
    assert set(curves.keys()) == {"demand", "uniform"}
    xs, ys = curves["demand"]
    assert xs == [50.0, 100.0]
    assert ys == [27.0, 28.5]


def test_load_pareto_rows_parses_csv(tmp_path):
    csv = tmp_path / "pareto.csv"
    csv.write_text(
        "arm,factor,anchors,PSNR,SSIM,LPIPS,n\n"
        "demand,1,100,28.5000,0.8300,0.1700,3\n"
    )
    rows = plot_pareto.load_pareto_rows(str(csv))
    assert rows[0]["arm"] == "demand"
    assert rows[0]["anchors"] == 100.0
    assert rows[0]["PSNR"] == 28.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/test_plot_pareto.py -v`
Expected: FAIL with `FileNotFoundError` / `ModuleNotFoundError` (plot_pareto.py does not exist).

- [ ] **Step 3: Write the plot script**

Create `scripts/plot_pareto.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/test_plot_pareto.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_pareto.py scripts/test_plot_pareto.py
git commit -m "feat: plot_pareto.py — demand vs uniform Pareto money figure"
```

- [ ] **Step 6: Hand the collate+plot to the user (after the sweep)**

Once Task 2's sweep is complete, the user runs:

```bash
python scripts/collect_results.py pareto --root /root/autodl-tmp/expA/amsterdam \
    --arms demand uniform --factors 0.25 0.5 1 2 \
    --step 25000 --checkpoint 30000 --output /root/autodl-tmp/expA/amsterdam/pareto.csv
python scripts/plot_pareto.py --csv /root/autodl-tmp/expA/amsterdam/pareto.csv \
    --metric PSNR --output /root/autodl-tmp/expA/amsterdam/pareto_psnr.png
```

Expected (paste back): the CSV with one row per feasible (arm, factor) and a PNG with two curves. Read the result: does demand lie on/above uniform, most at the low-budget (left) end? Report the per-factor ΔPSNR honestly — a tie is a legitimate Secondary outcome (Primary + efficiency still carry the paper).

---

### Task 6: Held-out test confirmation at the locked point

The swept curve is train-set. At the locked operating point (default `1×`, both arms), render held-out **test** from the saved checkpoints (no retrain) and run the near/far view split — confirming the demand-vs-uniform comparison survives on test, and recording the test numbers with the appearance caveat.

**Files:** none (uses existing `ocbgs/render.py`, `ocbgs/metrics.py`, `ocbgs/eval_view_split.py`).

- [ ] **Step 1: Hand the test-eval to the user**

For the locked factor (`1x`) and each arm, on seed_0:

```bash
for arm in demand uniform; do
  M=/root/autodl-tmp/expA/amsterdam/$arm/arm_1x/seed_0
  python ocbgs/render.py -m "$M" --eval --skip_train --ape -1
  python ocbgs/metrics.py -m "$M"
  echo "== $arm test view-split =="
  python ocbgs/eval_view_split.py -m "$M" --split test
done
```

- [ ] **Step 2: Review the test confirmation (review checkpoint)**

Expected (paste back): per-arm test PSNR/SSIM/LPIPS and the near/far split. The check is **demand vs uniform on test** at matched factor — does demand hold its train-set standing on held-out views? Record absolute test PSNR with the appearance caveat (`--ape -1`, shifted train-uid embeddings — approximate but fair across arms). Do not mark done without the pasted, completed output.

---

### Task 7: Reconcile eval-plan.md with the retired matched-budget arm

The spec §6 consistency edits to `docs/eval-plan.md` are still pending: Exp 1's matched-budget arm and the "equal #anchors" framing must be re-stated as "comparison along the Pareto curve at matched **achieved** #anchors, demand vs uniform", and the uniform control must be named as the flattened-demand controller arm (not only native Octree-GS). English doc edits.

**Files:**
- Modify: `docs/eval-plan.md` (§3 Exp 1 line 28; §1 line 14; §3 ablation "uniform = Octree-GS" line 35)

- [ ] **Step 1: Retire the matched-budget arm in Exp 1**

In `docs/eval-plan.md`, replace the Exp 1 "Matched-budget" bullet (line 28):

```markdown
- **Matched-budget:** force equality `Σn ≡ B_total = Octree-GS final #anchors` (plateau off; floor fills the budget). Strictly equal #anchors → "same budget, higher quality."
```

with:

```markdown
- **Matched-budget: RETIRED.** The force-equality / floor-fills-the-budget operating point is withdrawn — it contradicted the no-force-fill invariant (ADR-0003/0004/0005) and was experimentally refuted. The Pareto comparison is made at matched **achieved** #anchors along the swept curve (`docs/superpowers/specs/2026-06-26-controllable-budget-pareto-design.md` §6). See `--demand_uniform` for the uniform control arm.
```

- [ ] **Step 2: Re-state the critical comparison**

In `docs/eval-plan.md`, replace line 14:

```markdown
The critical comparison is vs Octree-GS at equal #anchors, isolating the demand-reallocation variable.
```

with:

```markdown
The critical comparison is **demand vs uniform allocation at matched achieved #anchors along the Pareto curve**, both run under identical controller machinery (uniform = the `--demand_uniform` flattened-demand arm; native Octree-GS is one reference point). This isolates the demand-reallocation variable without a force-fill matched-budget point.
```

- [ ] **Step 3: Name the uniform control mechanism in the ablation**

In `docs/eval-plan.md` §3 Demand-source ablation (line 35), change `uniform (= Octree-GS)` to `uniform (= the --demand_uniform flattened-demand controller arm; matches Octree-GS's spatial distribution)`.

- [ ] **Step 4: Commit**

```bash
git add docs/eval-plan.md
git commit -m "docs: reconcile eval-plan with retired matched-budget arm + --demand_uniform control (spec §6)"
```

---

## Out of scope (separate downstream plans)

These are graduation-relevant but independent of the Exp A money figure and are deferred to their own plans so this one produces a coherent, testable deliverable:

- **Exp A2 (train-under-budget vs prune-after-budget).** Overlays the OCB-3DGS-HR M3 render-time post-hoc curve (`D:\01_Projects\Active\Paper-research-1\OCB-3DGS-HR`, different gsplat stack) — a cross-repo integration warranting its own plan.
- **Exp A3 / C (unification + efficiency reporting).** FPS, active (opacity-masked) Gaussian count, training time on the same anchor population. Rides on the Exp A runs but needs a verification pass on what `render_sets`/`evaluate` already persist before tooling it — its own plan.
- **Exp D (continuity, stretch).** CLoD-GS continuous opacity decay + flicker/temporal-LPIPS. Explicitly out of the graduation-critical path.
- **Multi-scene extension.** quebec/rome repeat of this plan once the amsterdam double-curve is clean.

## Self-Review

- **Spec coverage:** Exp A (money figure) → Tasks 1–5; uniform-curve mechanism (spec §4 open decision a) → Task 1; eval protocol (spec §4 open decision, train+test) → Tasks 2 & 6; view split (Exp B confirm) → Task 6; relax set-point hit (spec §3) → Tasks 1/3; doc reconciliation (spec §6) → Task 7. A2/A3/C/D deferred with rationale. No gap in the Exp A core.
- **Placeholders:** none — every code step shows the full code; every GPU step gives the exact command + expected output + the no-bless-without-output rule.
- **Type consistency:** `resolve_controller_demand(d_a, uniform)` used identically in Task 1 helper and the gaussian_model call; `aggregate_pareto_points` row keys (`anchors_mean`, `PSNR_mean`, …) match between Task 4 producer and test; `split_curves`/`load_pareto_rows` keys (`arm`, `anchors`, `PSNR`) match between Task 5 producer (CSV columns from Task 4) and test.
