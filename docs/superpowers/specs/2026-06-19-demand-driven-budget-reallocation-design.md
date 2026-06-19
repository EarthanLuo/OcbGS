# Demand-Driven Budget Reallocation for LOD 3D Gaussian Splatting

**Status:** Design approved, ready for implementation planning
**Date:** 2026-06-19
**Working name:** DDBR (Demand-Driven Budget Reallocation)

## 1. Motivation

Existing LOD-structured 3DGS methods (e.g., Octree-GS) decouple rendering and
training: rendering selects a level of detail per view, but **training still
optimizes every Gaussian uniformly**, with densification driven by per-Gaussian
gradient magnitude and **no global budget constraint** (the model simply grows).
The render side knows where detail matters; the train side does not consume that
signal.

This project inserts a **middle controller** that feeds the render side's detail
demand back into training, so capacity is concentrated where detail is needed and
withdrawn where it is not. The headline goal: **at a fixed Gaussian budget,
achieve higher rendering quality than uniform allocation** (with reduced training
compute as a natural by-product).

The work is an explicit synthesis ("stitching") of four reference repositories:
`gsplat` (rasterizer backbone), `Octree-GS` (octree LOD structure + anchor
growing/pruning), `CLoD-GS` (continuous LOD via distance-based opacity decay),
and `FastGS` (multi-view-consistency importance scoring). The novel glue is the
**closed loop**: a pluggable demand field → a budget-conserving controller →
the existing growing/prune actuator.

### Future extension (out of scope for this paper)

A later paper replaces the demand producer with a **semantic / instance ROI
extractor**: "the detail I care about is this object/semantic class." Because the
demand producer is a pluggable interface emitting a per-voxel weight field, the
semantic version reuses the entire downstream pipeline unchanged. This design
constraint — demand as a swappable producer of a spatial weight field — is
honored throughout.

## 2. Core Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Demand signal source | **Error/visibility-driven** demand field, **normalized under a global budget** (rather than camera-geometry-only or externally-fixed budget) |
| D2 | Conserved budget | **Capacity** (total #Gaussians/anchors); compute savings reported as a by-product, not the core claim |
| D3 | Architecture | **Octree-GS as the spine**, absorbing CLoD-GS (render-side continuous opacity decay) and FastGS (importance scoring) parts as needed |
| D4 | Demand error term | **Gradient accumulator as primary** (free per-iter), **FastGS photometric residual as refinement** (periodic) |
| D5 | Budget conservation | **Hard constraint**, two-phase (ramp then steady-state reallocation) |
| D6 | Evaluation scale | **Standard benchmarks + one large-scale highlight scene** |
| D7 | Repo strategy | **Fork Octree-GS as project body** (`ocbgs/`); reference repos remain submodules for reference only |

## 3. System Architecture

The Octree-GS training loop (render → loss → backprop → optimize anchors) is kept
intact. We **replace only its densification policy**: the single global
gradient-threshold-driven `anchor_growing`/`prune` becomes a scheduling layer that
reads a demand field, enforces a global budget, and issues per-voxel grow/prune
targets.

```
                ┌─────────────────────────────────────────────┐
   every iter   │  Octree-GS training loop (render→loss→bp)     │
                └───────────────┬─────────────────────────────┘
                                │ record per-voxel error × visibility
                                ▼
                    ① DemandProducer (pluggable interface)
                       produce(scene, stats) → demand[voxel] = score
                       now:    ErrorVisibilityDemand
                       future: SemanticDemand  (same shape, drop-in)
                                │ demand field
                                ▼
   every N iter      ② BudgetController (pure function, unit-testable)
                       plan(demand, capacity, B_total) → ReallocationPlan
                       c*(v) = clamp(B_total · normalize(demand), floor, cap)
                       Δ(v) = c*(v) − n(v)
                       invariant: Σ n(v) ≡ B_total  (steady state)
                                │ per-voxel grow/prune targets
                                ▼
                    ③ Actuator = Octree-GS anchor_growing/prune
                       global threshold → per-voxel quota from controller
                                │
                                ▼
                       anchors reallocated → back to training loop
```

### Unit boundaries (rationale)

- **① DemandProducer** is an interface so the future semantic paper swaps the
  implementation with zero changes to the skeleton.
- **② BudgetController** is a **pure function** (inputs: demand + current capacity
  + budget; output: a reallocation plan). The conservation invariant
  `Σ n(v) ≡ B_total` is therefore verifiable in isolation by unit tests, with no
  CUDA dependency — this also enables local development on Windows (see §7).
- **③ Actuator** reuses Octree-GS's working `anchor_growing`/`prune_anchor`,
  parameterizing the single global threshold into per-voxel quotas — minimal
  intrusion.

The only essential difference from Octree-GS: densify/prune changes from
*gradient-driven, unbounded* to **demand-driven, budget-conserving, per-voxel
scheduled**.

## 4. Demand Field

### 4.1 Per-anchor raw signal

`s(a) = error(a) × visibility(a)`, computed from accumulators Octree-GS already
maintains per iteration in `training_statis` (near-zero added cost):

- **visibility(a)** = `anchor_demon` (number of views that observed the anchor).
- **error(a)** = **primary source A**: `offset_gradient_accum / offset_denom`
  (per-anchor mean gradient, already accumulated every iteration).
  - Caveat: gradient is a "should-I-densify" signal, not pure photometric error.
- **Refinement source B**: FastGS `compute_gaussian_score_fastgs` produces
  `pruning_score = photometric_loss × accum_loss_counts` — true error × observation
  count over a sampled camera set. Used **only at the periodic controller step**
  (every N iters) to correct the gradient-based demand. Cost amortized; also an
  ablation axis.

### 4.2 Aggregation to per-cell demand

**Octree structure recap.** Octree-GS samples anchors at *every* level
`cur_level ∈ [0, levels)` on a grid of size `voxel_size / fork^cur_level`.
Anchors at all levels **coexist**; a fine cell is one of the `fork³` children of
its parent coarse cell. Rendering activates a subset of levels per view by camera
distance (closer ⇒ finer levels). The detail capacity of a region is therefore
**how deeply that region is populated** (how many fine-level anchors it holds).

**The control unit is NOT only the coarsest (level-0) cell.** Demand and capacity
live across the hierarchy. We make the allocation granularity an explicit knob:

- **`control_level`** — the octree level whose cells the controller buckets demand
  into and allocates capacity over. Cells at a single level form a clean
  **non-overlapping partition** (required for budget normalization and the
  conservation invariant). Default: a mid-to-fine level, **not** level 0; tuned as
  an ablation (coarser `control_level` ⇒ coarser reallocation + less overhead/jitter;
  finer ⇒ finer control + more cells to manage).
- **Per-cell demand** `d(v) = Σ_{a ∈ column(v)} s(a)`, summing the per-anchor
  signal over all anchors (at any level) whose position falls inside control-cell
  `v`'s spatial column.
- **Capacity within a cell is realized at finer levels.** "Reallocating Gaussians
  from low-demand to high-demand cells" means: low-demand cells stop growing
  deeper / get their fine-level anchors pruned (coarsen), high-demand cells grow
  deeper (subdivide). Capacity = occupied octree depth, controlled per cell.

Throughout this document "voxel" is shorthand for a control-cell at `control_level`.

### 4.3 Budget normalization (the C part)

`c*(v) = clamp(B_total · d(v) / Σ d, floor, cap)`

- **floor**: every visible cell keeps a baseline capacity (prevents low-demand
  regions from being starved into mush).
- **cap**: prevents a single cell from monopolizing the budget.
- `Σ c*(v) ≈ B_total` — the controller's conservation target.

### 4.4 Cadence and smoothing

- Accumulate every iter (reuse `training_statis`, free).
- Recompute demand field + run controller every **N iters** (aligned with
  Octree-GS `check_interval = 100`).
- **EMA** smoothing of the demand field across windows to damp discrete
  grow/prune jumpiness.

### 4.5 Pluggable interface

```python
class DemandProducer:
    def produce(self, scene, stats) -> Tensor:  # shape [num_voxels]
        ...

# now:    ErrorVisibilityDemand (§4.1–4.2)
# future: SemanticDemand (semantic mask → per-voxel ROI weight, same shape)
```

## 5. Budget Controller

A pure function with a hard conservation guarantee.

**Inputs:** demand field `d(v)` (A smoothed + periodic B correction), current
per-voxel capacity `n(v)`, global budget `B_total`.

**Step 1 — surplus/deficit**

```
c*(v) = clamp(B_total · d(v)/Σd, floor, cap)   # target capacity
Δ(v)  = c*(v) − n(v)                            # >0 wants more, <0 has surplus
```

**Step 2 — translate to per-voxel actuator parameters**

- **Grow (Δ>0):** lower `anchor_growing`'s global `grad_threshold` per voxel —
  larger deficit ⇒ lower threshold (grows more aggressively); cap new anchors per
  voxel at `Δ(v)`.
- **Prune (Δ<0):** in surplus voxels, prune the `|Δ(v)|` lowest-`s(a)` anchors
  (reusing the demand signal for ranking; no new metric introduced).

**Step 3 — hard conservation, two-phase**

- **Phase 1 — ramp (`iter < T_budget`):** total grows from initial toward
  `B_total` with no forced pruning (fill the budget first).
- **Phase 2 — steady state (`iter ≥ T_budget`):** **prune surplus to free P
  slots, then redistribute exactly P slots to deficit voxels proportional to
  Δ⁺.** Total is pinned exactly at `B_total`.

⇒ `Σ n(v) ≡ B_total` holds literally and is unit-testable. Phase split avoids
starving early training.

**Step 4 — stability (anti-thrash)**

- **Dead-band:** ignore `|Δ(v)|` below a threshold (avoid anchors cycling in/out).
- **Rate limit:** move at most `k%` of `B_total` per controller step (e.g. 5%) for
  smooth convergence.
- EMA (§4.4) keeps the demand field itself from jumping.

**Setting `B_total`:**

- Main experiments: `B_total = #Gaussians at Octree-GS convergence` → claim
  "same budget, higher quality".
- Sweep `B_total` over several values → quality-vs-#Gaussians Pareto curve (the
  money figure).

## 6. Evaluation Plan

### 6.1 Baselines

| Method | Role |
|--------|------|
| Vanilla 3DGS (gsplat) | capacity-agnostic reference |
| **Octree-GS** | **primary base / control** (= uniform allocation; ours = + demand reallocation) |
| CLoD-GS | continuous-LOD comparison |
| FastGS | training-speed axis (optional) |

The critical comparison is **vs Octree-GS at equal #Gaussians**, isolating the
demand-reallocation variable.

### 6.2 Metrics

- Quality: PSNR / SSIM / LPIPS.
- Budget/efficiency: **#Gaussians (controlled variable)**, memory, render FPS,
  training time (the "less compute" by-product).

### 6.3 Experiments

1. **Main table:** `B_total = Octree-GS converged #Gaussians`, compare quality.
   Claim: same budget, higher quality.
2. **Money figure — Pareto curve:** sweep `B_total ∈ {0.25, 0.5, 1, 2}× baseline`
   → quality. Claim: our curve dominates across budgets.
3. **Ablations:**
   - Demand source: uniform (= Octree-GS) / A only (gradient) / A+B (gradient +
     photometric).
   - Conservation: hard / soft / none.
   - Control granularity: `control_level` (coarse → fine) — reallocation
     granularity vs overhead/jitter trade-off.
   - Controller knobs: floor/cap, EMA, rate-limit `k` sensitivity.
4. **By-product:** training time / FPS (the compute-saving corollary).
5. **Qualitative:** per-voxel capacity heatmap showing capacity flowed to
   high-detail regions — also previews the future semantic version (swap heatmap
   for semantic ROI).

### 6.4 Datasets (Decision D6: standard + one large highlight)

- Standard: Mip-NeRF360, Tanks & Temples, Deep Blending — the expected benchmark
  tables (main table + Pareto + ablations).
- One large-scale highlight scene (MatrixCity aerial *or* a BungeeNeRF multi-scale
  scene) — a single "scales to large scenes" figure/small table, where the
  budget-scarcity story is strongest.

## 7. Engineering & Environment

### 7.1 Development / execution split

- **Development environment:** Windows 10 + RTX 3060 (local).
- **Execution environment:** Linux + arbitrary GPU count + arbitrary PyTorch
  version (rented servers, group-reimbursed).

Octree-GS / CLoD-GS depend on **custom CUDA rasterizer submodules** that are
painful to build on Windows. We **never run full training locally**. This aligns
with the §3 design:

- **Local (Windows + 3060):** run the pure-logic layer + **unit tests** (feed the
  controller synthetic tensors; verify the conservation invariant
  `Σ n ≡ B_total`, demand aggregation, surplus/deficit math). **No custom CUDA
  build** — the rasterizer import is lazy/optional so core modules do not depend
  on it transitively.
- **Server (Linux):** `conda` env, compile Octree-GS CUDA submodule, run full
  training and all experiments.

Dev loop: write logic + unit-test locally → push → pull & run experiments on
server. Local work is never blocked by CUDA compilation.

### 7.2 GPU usage

Octree-GS-class 3DGS training is **single-GPU** (no data parallelism). "Arbitrary
GPU count" therefore means **job-level parallelism**: one (scene × budget)
combination per GPU, fanning the Pareto sweep and multi-scene runs across GPUs.
The experiment queue is organized accordingly.

### 7.3 Reproducibility

- `environment.yml` (loose pins, tolerant of the server's arbitrary PyTorch
  version) + a one-shot `setup.sh` (create env, build submodule).
- Fixed random seeds.
- Octree-GS `arguments/` config system records every experiment's settings.

### 7.4 Repo layout (Decision D7: fork Octree-GS as body)

- `ocbgs/` — a fork of the Octree-GS codebase as the working baseline, with
  **minimal-intrusion** edits to `gaussian_model.py` / `train.py` to call the
  controller, plus two new isolated modules:
  - `ocbgs/demand/` — `DemandProducer` interface + `ErrorVisibilityDemand`.
  - `ocbgs/controller/` — the pure-function `BudgetController` + plan types.
- `refered_repo/` — reference submodules, **for reference only**.
- Trade-off accepted: no automatic upstream sync with Octree-GS (irrelevant for a
  synthesis paper).

**Directory & module hygiene (hard requirement).** The structure must stay clean:
each module has one clear responsibility and a visible boundary; no scattered
"functions flying everywhere"; no oversized single files (split when a file grows
past a reasonable length / does more than one thing); shallow, explicit coupling
(`demand/` and `controller/` depend on neither each other's internals nor the CUDA
rasterizer). This is a standing constraint on all code in this project, not a
one-off cleanup.

## 8. Out of Scope

- **Interactive viewer.** Paper-stage presentation is **offline-rendered images
  only** (figures/comparisons produced by the eval `render.py` + the §6.3 capacity
  heatmap). No interactive viewer work is needed for the paper; Octree-GS's SIBR
  desktop viewer is available if ad-hoc inspection helps, but is not a deliverable.
- **Potree-style web / streaming viewer** (octree-streamed, click-to-highlight) —
  a future engineering milestone, naturally paired with the semantic-ROI paper;
  not part of this work.
- Semantic / instance-driven demand producer (future paper; interface reserved).
- Multi-GPU distributed training (single-GPU per job by design).
- Differentiable/continuous capacity actuator (CLoD-GS opacity decay used only on
  the render side; capacity changes are discrete grow/prune).
