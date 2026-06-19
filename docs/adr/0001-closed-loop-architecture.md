# ADR-0001: Closed-loop architecture & module boundaries

**Status:** Accepted
**Source spec:** `docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`

## Context

Existing LOD-structured 3DGS methods (e.g., Octree-GS) decouple rendering and training: rendering selects a level of detail per view, but training still optimises every Gaussian uniformly, with densification driven by per-Gaussian gradient magnitude and no global budget constraint (the model simply grows). The render side knows where detail matters; the train side does not consume that signal.

This project inserts a middle controller that feeds the render side's detail demand back into training, so capacity is concentrated where detail is needed and withdrawn where it is not. The headline goal: at a fixed Capacity Budget, achieve higher rendering quality than uniform allocation (with reduced training compute as a natural by-product).

The work is an explicit synthesis of four reference repositories: `gsplat` (rasterizer backbone), `Octree-GS` (octree LOD structure + anchor growing/pruning), `FastGS` (multi-view-consistency importance scoring), and `CLoD-GS` (continuous LOD via distance-based opacity decay — baseline, not a core component). The novel glue is the closed loop: a pluggable demand field → a budget-conserving controller → the existing grow/prune actuator.

## Decision

### D1 — Demand signal is error/visibility-driven, normalised under the Capacity Budget

The demand field is built from error and visibility signals rather than camera-geometry alone or an externally-fixed budget. This is the essence of the closed loop: the render side's observed error tells the controller where capacity is needed, and the Capacity Budget normalisation makes allocations comparable across Control Cells.

### D3 — Octree-GS spine + FastGS importance; CLoD-GS is a baseline

The architecture is built on the Octree-GS training loop (render → loss → backprop → optimise anchors), replacing only its densification policy. FastGS contributes per-anchor importance scoring as a periodic refinement signal. CLoD-GS is a baseline, not a core component; its render-side opacity decay is an optional, orthogonal, render-only ablation that barely moves still-image metrics.

### Four-unit pipeline

The Octree-GS training loop is kept intact. We replace only its densification policy: the single global gradient-threshold-driven `anchor_growing`/`prune` becomes a scheduling layer that reads a demand field, enforces the Budget Constraint, and issues per-Control-Cell grow/prune targets.

```
               ┌─────────────────────────────────────────────┐
  every iter   │  Octree-GS training loop (render→loss→bp)   │
               └───────────────┬─────────────────────────────┘
                               │ training_statis (+ FastGS photometric)
                               ▼
           ① DemandProducer (pluggable; partition-agnostic)
              produce(stats) → s(a): Anchor Demand  [N_anchors]
                               │ s(a)
               ┌───────────────┴──────────────────────────────┐
               ▼                                              │
  ② Partition (pure reduce, no CUDA)                          │ s(a)
     d(v) = segment_sum(s(a), Cell Membership)                │ (which-to-prune)
     owns control_level + membership                          │
               │ d(v): Demand Field  [N_cells]                │
               ▼                                              │
  every N iter ③ BudgetController (pure fn, unit-testable)   │
     plan(d, occupancy, B_total) → ReallocationPlan           │
     decides HOW MANY per cell (grow quota / prune count)     │
               │ ReallocationPlan (per-cell counts)           │
               ▼                                              ▼
           ④ Actuator = Octree-GS anchor_growing/prune (pure PyTorch)
              grow: cap each cell's candidates to grow quota (by gradient)
              prune: remove surplus lowest-s(a) — decides WHICH, executes
                               │
                               ▼
                      anchors reallocated → back to training loop
```

Unit responsibilities:

- **① DemandProducer** — turns raw stats into per-anchor Anchor Demand `s(a)`, and nothing else. Partition-agnostic (knows no Control Cell/`control_level`).
- **② Partition** — owns Cell Membership + `control_level` and the pure reduction `s(a) → d(v)` (segment-sum). The partition unit is the **Control Cell**: an occupied octree cell at the `control_level`. CUDA-free.
- **③ BudgetController** — a pure function (inputs: Demand Field + Cell Occupancy + Capacity Budget; output: a plan of per-Control-Cell counts). Never touches `s(a)`; works purely at Control-Cell level.
- **④ Actuator** — reuses Octree-GS's working `anchor_growing`/`prune_anchor` (both pure PyTorch). Controller decides *how many*, Actuator decides *which*.

The only essential difference from Octree-GS: densify/prune changes from gradient-driven, unbounded to demand-driven, budget-conserving, per-Control-Cell scheduled.

### Three hard constraints

1. **`s(a)` is computed once and fans out to two consumers** — Partition's reduction (`s(a) → d(v)`) and the Actuator's prune ranking (lowest-`s(a)`).

2. **Controller never touches `s(a)`** — it works purely at Control-Cell level (Demand Field `d(v)` → ReallocationPlan). The per-anchor signal is opaque to the controller; this keeps the controller independent of the demand producer and unit-testable without anchors.

3. **DemandProducer is partition-agnostic** — it knows nothing of Control Cells, `control_level`, or the Capacity-Budget-derived partition. This is the load-bearing constraint that enables the future semantic producer: a `SemanticDemand` that emits per-anchor scores from a semantic mask drops in with zero skeleton change and no dependency on the budget-derived `control_level`.

### CUDA rasterizer is never modified

Every change — demand, partition, controller, actuator, and any optional CLoD-GS opacity decay — is Python-side. The render path passes opacity as a tensor argument to the stock `diff_gaussian_rasterization` submodule. This preserves the CUDA-free / locally-testable invariant end to end.

## Consequences

### Repo layout (D7)

The project forks Octree-GS as the working baseline (`ocbgs/`), with minimal-intrusion edits to `gaussian_model.py` / `train.py` to call the controller and apply the plan, plus three new isolated, CUDA-free modules:

- `ocbgs/demand/` — `DemandProducer` interface + `ErrorVisibilityDemand`; emits per-anchor Anchor Demand `s(a)`. Partition-agnostic.
- `ocbgs/partition/` — Cell Membership + `control_level` derivation + the pure `s(a) → d(v)` segment-sum reduction. Reused verbatim by future producers. Membership is stateless.
- `ocbgs/controller/` — the pure-function `BudgetController` + plan types; Control-Cell level only (decides *how many*).

The Actuator lives as minimal-intrusion hooks in the forked `gaussian_model.py` (it must mutate anchor state and decides *which* to prune via `s(a)`). Reference repositories stay under `refered_repo/` as submodules, for reference only.

### Development / execution split

- **Local (Windows + RTX 3060):** run the pure-logic layer + unit tests. No custom CUDA build — the rasterizer import is lazy/optional so core modules do not depend on it transitively. Local work is never blocked by CUDA compilation.
- **Server (Linux):** compile the Octree-GS CUDA submodule, run full training and all experiments.

Dev loop: write logic + unit-test locally → push → pull & run experiments on server.

### Directory hygiene (hard requirement)

Each module has one clear responsibility and a visible boundary. No scattered functions; no oversized single files; split when a file grows past a reasonable length or does more than one thing. Shallow, explicit coupling — `demand/` and `controller/` depend on neither each other's internals nor the CUDA rasterizer.

### `adjust_anchor` integration

The controller is embedded by rewriting the body of `adjust_anchor` (not an external wrapper). The native training loop call site in `train.py` is unchanged. The pure logic lives in `ocbgs/partition` + `ocbgs/controller` (unit-tested); `gaussian_model.py` becomes a thin orchestrator + tensor-mutation seam.

## Non-goals

- **Differentiable/continuous capacity actuator.** Capacity changes are discrete grow/prune. CLoD-GS's continuous opacity decay is a render-only, orthogonal, optional ablation — not a differentiable capacity mechanism, not a core component.
- **CLoD-GS as a core component.** CLoD-GS is a baseline only. Its opacity decay, if applied, is a one-line Python multiply in the eval render path, never inside `training_statis`.
- **Semantic / instance-driven demand producer.** The `SemanticDemand` is future work; the interface is reserved and the architecture is designed to accept it, but its implementation is out of scope.
- **Multi-GPU distributed training.** Training is single-GPU per job by design. "Arbitrary GPU count" means job-level parallelism: one (scene × budget) combination per GPU.
- **Interactive viewer.** Paper-stage presentation is offline-rendered images only. Octree-GS's SIBR desktop viewer is available for ad-hoc inspection but is not a deliverable. Potree-style web/streaming viewer is a future engineering milestone.
