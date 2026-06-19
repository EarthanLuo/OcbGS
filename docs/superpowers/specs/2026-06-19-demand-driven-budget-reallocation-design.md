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
demand producer is a pluggable interface emitting a per-cell weight field, the
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
| D5 | Budget conservation | **Hard upper-bound constraint** (`Σn ≤ B_total`), two emergent phases (ramp then steady-state reallocation) |
| D6 | Evaluation scale | **Standard benchmarks + one large-scale highlight scene** |
| D7 | Repo strategy | **Fork Octree-GS as project body** (`ocbgs/`); reference repos remain submodules for reference only |

## 3. System Architecture

The Octree-GS training loop (render → loss → backprop → optimize anchors) is kept
intact. We **replace only its densification policy**: the single global
gradient-threshold-driven `anchor_growing`/`prune` becomes a scheduling layer that
reads a demand field, enforces a global budget, and issues per-cell grow/prune
targets.

```
                ┌─────────────────────────────────────────────┐
   every iter   │  Octree-GS training loop (render→loss→bp)     │
                └───────────────┬─────────────────────────────┘
                                │ record per-cell error × visibility
                                ▼
                    ① DemandProducer (pluggable interface)
                       produce(scene, stats) → demand[cell] = score
                       now:    ErrorVisibilityDemand
                       future: SemanticDemand  (same shape, drop-in)
                                │ demand field
                                ▼
   every N iter      ② BudgetController (pure function, unit-testable)
                       plan(demand, capacity, B_total) → ReallocationPlan
                       c*(v) = clamp(B_total · normalize(demand), floor, cap)
                       Δ(v) = c*(v) − n(v)
                       invariant: Σ n(v) ≡ B_total  (steady state)
                                │ per-cell grow/prune targets
                                ▼
                    ③ Actuator = Octree-GS anchor_growing/prune
                       global threshold → per-cell quota from controller
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
  parameterizing the single global threshold into per-cell quotas — minimal
  intrusion.

The only essential difference from Octree-GS: densify/prune changes from
*gradient-driven, unbounded* to **demand-driven, budget-conserving, per-cell
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

**Fusion scale alignment (A+B only).** When B refines A (e.g. `s = g ⊙ (1 + α·p̂)`
or `g + λ·p`), the two signals are put on a comparable scale *at the fusion point*
(e.g. each normalized to unit sum) so the mixing weight is meaningful and one does
not swamp the other. This scale alignment lives in the controller's fusion step,
**not** in the Demand Producer contract — a single-source producer needs none of
it, and forcing per-producer distribution alignment would distort the very signal
shape the ablation measures (see §6.3 fairness condition).

### 4.2 Aggregation to per-cell demand

**Octree structure recap.** Octree-GS samples anchors at *every* level
`cur_level ∈ [0, levels)` on a grid of size `voxel_size / fork^cur_level`.
Anchors at all levels **coexist**; a fine cell is one of the `fork³` children of
its parent coarse cell. Rendering activates a subset of levels per view by camera
distance (closer ⇒ finer levels). The detail capacity of a region is therefore
**how deeply that region is populated** (how many fine-level anchors it holds).

**The control unit is NOT only the coarsest (level-0) cell.** Demand and capacity
live across the hierarchy. (Terminology for this section is fixed in `CONTEXT.md`;
"voxel" is retired in favour of **Control Cell**.)

- **Control Cell / Control Level.** A Control Cell is an *occupied* octree cell at
  the `control_level` — a box becomes a Control Cell once it holds an anchor or
  carries demand; empty space is not a Control Cell. Cells at a single level form a
  **non-overlapping partition**, required for budget normalization and the Budget
  Constraint.
- **Cell Membership (by position).** Each anchor belongs to exactly one Control
  Cell: the one whose box contains the anchor's position, **independent of the
  anchor's own level**. A coarse anchor is billed to the single cell containing its
  centre (no area-weighted splitting — fractional anchors would break both the
  partition and the grow/prune actuator). Its cross-cell influence is recovered
  through the demand channel, not the membership channel.
- **Per-cell demand** `d(v) = Σ_{a : member(a)=v} s(a)`.
- **Capacity within a cell is realized at finer levels.** "Reallocating Gaussians
  from low-demand to high-demand cells" means: low-demand cells stop growing
  deeper / get their fine-level anchors pruned (coarsen), high-demand cells grow
  deeper (subdivide).

**`control_level` is derived, not a free knob.** Its feasible range is bounded on
both ends by the floor (§4.3) and the resulting active-cell count
`N_active(level)` (number of occupied Control Cells at that level):

- *Too fine* ⇒ `N_active → #anchors → B_total` ⇒ reallocatable slack
  `S = B_total − floor · N_active` collapses ⇒ the floor pins everyone ⇒ controller
  degenerates to identity.
- *Too coarse* ⇒ `N_active → 1` ⇒ nothing to reallocate between ⇒ degenerates to
  uniform.

Given a **reallocation-headroom fraction `τ`** (the share of the budget kept free
for actual movement) and a minimum cell count `A_min`, `control_level` is derived:

```
control_level = max { level :
                      floor · N_active(level) ≤ (1 − τ) · B_total      (feasibility + τ slack)
                      ∧ N_active(level) ≥ A_min }                       (enough cells to move between)
```

i.e. **the finest level that still leaves headroom `τ` and enough cells**. This
makes the granularity reproducible across datasets/budgets: fix `τ`, and
`control_level` falls out of `B_total` automatically. The ablation sweeps **`τ`**
(not raw `control_level`).

### 4.3 Budget normalization (the budget-constraint part)

`c*(v) = clamp(B_total · d(v) / Σ d, floor, cap)`

- The normalization `d(v)/Σ d` is **L1** (pure positive scaling): it removes
  absolute scale but preserves rank and all inter-cell ratios — i.e. it preserves
  the demand's *shape*, which is the signal's information. The Demand Score
  contract (§4.1) need only be non-negative and cross-cell-comparable; units never
  reach the controller.
- **floor**: a baseline Target Capacity that protects existing/observed content
  from being starved into mush. **Applies only to active Control Cells** (occupied
  or demand > 0); empty space gets no floor, so `Σ floor = floor · N_active`.
- **cap**: prevents a single cell from monopolizing the budget.
- Feasibility (see §4.2 derivation): `floor · N_active ≤ (1 − τ) · B_total` must
  hold, else the Budget Constraint is physically unsatisfiable.
- `Σ c*(v) ≈ B_total` here; §5 turns the `≈` into the exact Budget Constraint.

### 4.4 Cadence and smoothing

- Accumulate every iter (reuse `training_statis`, free).
- Recompute demand field + run controller every **N iters** (aligned with
  Octree-GS `check_interval = 100`).
- **EMA** smoothing of the demand field across windows to damp discrete
  grow/prune jumpiness.

### 4.5 Pluggable interface

```python
class DemandProducer:
    # Contract: return raw, non-negative, cross-cell-comparable Demand Scores in
    # [0, +inf), one per active Control Cell. No units, no per-producer
    # normalization — the single L1 normalization lives in the controller.
    def produce(self, scene, stats) -> Tensor:  # shape [N_active]
        ...

# now:    ErrorVisibilityDemand (§4.1–4.2)
# future: SemanticDemand (semantic mask → per-cell ROI weight, same contract)
```

## 5. Budget Controller

A pure function with a hard conservation guarantee.

**Inputs:** Demand Field `d(v)` (gradient, EMA-smoothed + periodic photometric
correction), current Cell Occupancy `n(v)`, Capacity Budget `B_total`.

**Step 1 — surplus/deficit**

```
c*(v) = clamp(B_total · d(v)/Σd, floor, cap)   # target capacity
Δ(v)  = c*(v) − n(v)                            # >0 wants more, <0 has surplus
```

**Step 2 — translate to per-cell actuator parameters**

- **Grow (Δ>0):** lower `anchor_growing`'s global `grad_threshold` per cell —
  larger deficit ⇒ lower threshold (grows more aggressively); cap new anchors per
  cell at `Δ(v)`.
- **Prune (Δ<0):** in surplus cells, prune the `|Δ(v)|` lowest-`s(a)` anchors
  (reusing the demand signal for ranking; no new metric introduced).

**Step 3 — hard conservation, two emergent phases (no tuned `T_budget`)**

The switch is **state-triggered, not a magic iteration count** (a fixed count
mis-fires both ways: too early prunes immature anchors, too late forces a huge
overshoot prune).

- **Phase 1 — demand-guided ramp:** total grows toward `B_total`, guided by demand,
  with no forced pruning. As `N_total` nears `B_total`, growth is **budget-aware**:
  the crossing step's grow quota is scaled by a single global factor
  `p = (B_total − N_total) / Σ Δ⁺` so the total lands exactly on `B_total` instead
  of overshooting (**proportional** clamp — fair, no sampling bias while the demand
  signal may still be immature). Once at `B_total`, `p → 0` naturally **freezes
  structure** so anchors keep training to maturity until the gate opens.
- **Phase switch (emergent):** enter Phase 2 when **`N_total ≥ B_total` AND the
  demand ranking has stabilized** — Spearman rank correlation of `d(v)` between
  consecutive controller windows, computed over their **shared** cells, `≥ 0.9`,
  **sustained for k (≈2–3) consecutive windows**. Rank stability (not EMA magnitude
  stability) is the correct gate: pruning ranks by `s(a)`, and ranks can hold while
  magnitudes drift under global rescaling. *Plateau fallback:* if growth flattens
  below `B_total`, enter Phase 2 under the cap (see budget semantics, §5 / §6.3).
- **Phase 2 — steady-state reallocation:** prune surplus to free P slots, then
  redistribute exactly P slots to deficit cells ∝ Δ⁺.

**Budget semantics.** The Budget Constraint is an **upper bound** `Σ n(v) ≤ B_total`,
not a strict equality. Plateau (the scene cannot productively use the full budget)
is left as honest slack rather than padded with low-value anchors that would only
hurt the reported FPS/memory.

⇒ **Three-part testable invariant** (one unit test asserts all three):
1. Phase-2 reallocation conserves exactly — `P_in = P_out`, no leak.
2. The total stays `Σ n(v) ≤ B_total` at all times.
3. When binding (capacity fully demand-justified), Phase-2 total `Σ n(v) ≡ B_total`.

`T_budget` is removed — the phase boundary is an emergent function of system state,
mirroring how `control_level` is derived (§4.2).

**Step 4 — stability (anti-thrash)**

- **Dead-band:** ignore `|Δ(v)|` below a threshold (avoid anchors cycling in/out).
- **Rate limit:** move at most `k%` of `B_total` per controller step (e.g. 5%) for
  smooth convergence. (With the budget-aware ramp there is no overshoot cliff at
  the phase switch, so the rate limit only governs steady-state churn.)
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

1. **Main comparison — two operating points, both points on the Pareto curve:**
   - **Matched-budget:** force equality `Σn ≡ B_total = Octree-GS converged
     #Gaussians` (plateau off; floor fills the budget). Strictly equal #Gaussians
     → "same budget, higher quality"; answers the "your N must match" objection.
   - **Natural-budget:** the cap `Σn ≤ B_total` (plateau allowed). #Gaussians `≤`
     baseline at higher quality → "less budget, higher quality" (stronger claim;
     #Gaussians reported explicitly).
2. **Money figure — Pareto curve:** sweep `B_total ∈ {0.25, 0.5, 1, 2}× baseline`
   → quality. Claim: our curve dominates across budgets.
3. **Ablations:**
   - Demand source: uniform (= Octree-GS) / gradient only / gradient + photometric.
     **Fairness condition:** hold the entire downstream pipeline identical across
     arms (EMA, cadence, controller L1 normalization, floor/cap) and vary *only*
     the raw signal source — so any quality delta is attributable to the signal's
     informativeness, not to an incidental change of distribution shape. (No
     per-producer distribution alignment; that would erase the shape the ablation
     measures.)
   - Conservation: hard / soft / none.
   - Reallocation headroom: sweep **`τ`** (`control_level` is derived from
     `τ` + `B_total`, §4.2) — granularity vs slack trade-off.
   - Controller knobs: floor/cap, EMA, rate-limit `k` sensitivity.
4. **By-product:** training time / FPS (the compute-saving corollary).
5. **Qualitative:** per-cell Target Capacity heatmap showing capacity flowed to
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
