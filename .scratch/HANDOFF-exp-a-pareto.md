# Handoff — exp-a-pareto: Implement Exp A (Controllable-Budget Pareto Front)

**Branch:** `exp-a-pareto` (base: `main` @ `b73d9c6`)
**Repo:** `https://github.com/EarthanLuo/OcbGS`
**Reviewer:** Claude (reviews each task after you push; gives go/no-go before you proceed)

## Your job

Implement the plan at `docs/superpowers/plans/2026-06-27-exp-a-controllable-budget-pareto.md` **task by task**, committing each task and pushing before the reviewer signs off. Do NOT batch tasks into one commit.

## Setup

```bash
git fetch origin
git checkout exp-a-pareto
cd <repo root>
# Python path for local tests:
export PYTHONPATH="$PWD/ocbgs:$PYTHONPATH"
```

## What is already done (do not re-derive)

- **Source B (photometric demand) is a tested NEGATIVE result.** Far-view ΔPSNR = −0.218 on held-out test. B is demoted; the Exp A sweep is **A-only (gradient demand)**. Do not add B to any sweep arm.
- **`--grow_relax_scale 0.1`** is the validated switch that lets the controller fill 99.7% of `B_total` on amsterdam. It is already wired in `ocbgs/arguments/__init__.py` and `ocbgs/scene/gaussian_model.py:1071`. Do not change its default (1.0); pass `--grow_relax_scale 0.1` explicitly on every controller training run.
- **The controller's `d_sum<=0 → B_total/N_active` branch already exists** (`ocbgs/controller/__init__.py:86-87`). Task 1 only needs to zero `d_a` before the `controller.plan()` call to trigger it — no changes to the controller internals.

## Task sequence and ownership

| Task | What | Where to run tests |
|------|------|--------------------|
| **1** | `--demand_uniform` flag + `resolve_controller_demand` helper | **Local** (pure-logic, CPU tensors) |
| **2** | amsterdam Pareto sweep script | Server (bash, no local test needed) |
| **3** | Relax fill-rate smoke (gate before the 24-run sweep) | Server |
| **4** | `collect_results.py pareto` subcommand | **Local** (pure-logic) |
| **5** | `plot_pareto.py` | **Local** (pure-logic; matplotlib only called in `main()`) |
| **6** | Held-out test confirmation + view-split | Server |
| **7** | `eval-plan.md` reconciliation | Local (doc edit) |

**Tasks 1, 4, 5:** write the failing test first, then the minimal implementation, then run the test to green. The plan has the exact test code — use it verbatim as the starting point.

**Tasks 2, 3, 6:** the plan has the exact server commands. Push the script/notes, then the human runs it on the GPU server and pastes the output back to the reviewer.

## Hard constraints (Global Constraints in the plan — no exceptions)

1. **A-only sweep.** `b_enabled` default is already `False`; do not pass any `--fusion_lambda` or B flags to the Pareto arms.
2. **No force-fill.** Never add a `--plateau_enabled False` or any floor-fill path to hit `B_total` exactly. The Pareto x-axis is *achieved* anchors, not set-point.
3. **`|| true` on every `(( counter++ ))`** in bash scripts. Bare `(( _running++ ))` with `set -e` kills the script on the first `0→1` increment (CLAUDE.md pitfall). Mirror `exp4_bungeenerf.sh` exactly.
4. **Per-seed resume guard** on every arm: `[ -f "$armdir/seed_$seed/results.json" ] && continue`. Re-running the script must never recompute a finished seed.
5. **`--ds 1` for amsterdam** (not `-i images`). The COLMAP reader ignores `-i`; it maps `ds=1 → images/`.
6. **Train-set metrics on the sweep** (no `--eval`). Held-out test is only rendered at the locked point in Task 6.
7. **`--grow_relax_scale 0.1` on every controller arm** (demand and uniform both). Omitting it gives ~73% fill, making the x-axis meaningless.

## Key file locations

| What | Path |
|------|------|
| Full plan | `docs/superpowers/plans/2026-06-27-exp-a-controllable-budget-pareto.md` |
| Controller (`_allocate`, `d_sum<=0` branch) | `ocbgs/controller/__init__.py:46–232` |
| Argument group (`grow_relax_scale` example) | `ocbgs/arguments/__init__.py:164–183` |
| Model wiring (`grow_relax_scale` example) | `ocbgs/scene/gaussian_model.py:449` |
| `controller.plan()` call site | `ocbgs/scene/gaussian_model.py:1313` |
| Existing sweep skeleton to adapt | `scripts/exp2_garden_pareto.sh` |
| Existing collator to extend | `scripts/collect_results.py` |
| B_total file (amsterdam, server) | `/root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam` |

## Commit and review protocol

1. Finish one task → `git add <files> && git commit -m "feat/docs: ..."` → `git push`
2. Notify the reviewer (paste the commit hash + a one-line summary).
3. Wait for go/no-go before starting the next task.
4. If the reviewer flags a 🔴 issue, fix it in a new commit on the same branch — do **not** amend a pushed commit.

## Local test command

```bash
# Task 1
python -m pytest ocbgs/tests/test_demand_uniform.py -v

# Task 4
python -m pytest scripts/test_collect_pareto.py -v

# Task 5
python -m pytest scripts/test_plot_pareto.py -v
```

All three test files must be written **before** the implementation (TDD: red first, then green).
