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
withdrawn where it is not. The headline goal: **at a fixed anchor budget,
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
demand producer is a pluggable, partition-agnostic interface emitting a per-anchor
demand signal, the semantic version reuses the entire downstream pipeline
(Partition → Controller → Actuator) unchanged. This design constraint — demand as
a swappable per-anchor producer, decoupled from the spatial partition — is honored
throughout.

## 2. Core Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Demand signal source | **Error/visibility-driven** demand field, **normalized under a global budget** (rather than camera-geometry-only or externally-fixed budget) |
| D2 | Conserved budget | **Capacity** (total #anchors; rendered Gaussians/memory derived); compute savings reported as a by-product, not the core claim |
| D3 | Architecture | **Octree-GS as the spine**, absorbing CLoD-GS (render-side continuous opacity decay) and FastGS (importance scoring) parts as needed |
| D4 | Demand error term | **Gradient accumulator as primary** (free per-iter), **FastGS photometric residual as refinement** (periodic) |
| D5 | Budget conservation | **Hard upper-bound constraint** (`Σn ≤ B_total`), two emergent phases (ramp then steady-state reallocation) |
| D6 | Evaluation scale | **Standard benchmarks (near-uniform control) + BungeeNeRF** as the large-scale highlight (chosen to maximize demand non-uniformity; MatrixCity dropped as near-uniform) |
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
                                │ training_statis (+ FastGS photometric)
                                ▼
            ① DemandProducer (pluggable; partition-agnostic)
               produce(stats) → s(a): Anchor Demand  [N_anchors]
               now: ErrorVisibilityDemand   future: SemanticDemand
                                │ s(a)
                ┌───────────────┴───────────────────────────┐
                ▼                                            │
   ② Partition (pure reduce, no CUDA)                        │ s(a)
      d(v) = segment_sum(s(a), Cell Membership)              │ (which-to-prune)
      owns control_level + membership                        │
                │ d(v): Demand Field  [N_cells]              │
                ▼                                            │
   every N iter ③ BudgetController (pure fn, unit-testable)  │
      plan(d, occupancy, B_total) → ReallocationPlan         │
      c*(v)=clamp(B_total·d/Σd, floor, cap); Δ(v)=c*(v)−n(v) │
      invariant: Σ n(v) ≤ B_total  (§5 three-part)           │
      decides HOW MANY per cell (grow quota / prune count)   │
                │ ReallocationPlan (per-cell counts)         │
                ▼                                            ▼
            ④ Actuator = Octree-GS anchor_growing/prune (pure PyTorch)
               grow: cap each cell's candidates to Δ(v) (by gradient)
               prune: remove |Δ(v)| lowest-s(a) — decides WHICH, executes
                                │
                                ▼
                       anchors reallocated → back to training loop
```

### Unit boundaries (rationale)

- **① DemandProducer** turns raw stats into per-anchor **Anchor Demand** `s(a)` —
  and nothing else. It is **partition-agnostic** (knows no cell/`control_level`),
  so the future semantic producer drops in with zero skeleton change and no
  dependency on the budget-derived `control_level`.
- **② Partition** owns Cell Membership + `control_level` and the pure reduction
  `s(a) → d(v)` (segment-sum). A CUDA-free `(positions, scores, membership) →
  per-cell sums` operation — unit-testable locally on Windows, reused verbatim by
  the semantic producer.
- **③ BudgetController** is a **pure function** (inputs: Demand Field + occupancy +
  budget; output: a plan of per-cell *counts*). Never touches `s(a)`; works purely
  at cell level. The three-part Budget Constraint invariant (§5) is verifiable in
  isolation, no CUDA — enabling local development on Windows (see §7).
- **④ Actuator** reuses Octree-GS's working `anchor_growing`/`prune_anchor` (both
  pure PyTorch): on grow it **caps each cell's proposed candidates to `Δ(v)`** (by
  proposing gradient) before materialization; on prune it consumes `s(a)` to pick
  the lowest-demand anchors. Responsibility split: Controller decides *how many*,
  Actuator decides *which* (§5).

`s(a)` is computed once and fans out to two consumers (Partition's reduction and
the Actuator's ranking); the Controller stays cell-level only.

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

**Fusion of A (gradient) and B (photometric).** B is a *second* per-anchor demand
signal (FastGS `pruning_score`), reduced to `d_B(v)` by the same Partition. The
controller fuses **additively** after per-signal scale alignment:

`d(v) = EMA_τsmooth[ normalize(d_A) + λ · normalize(d_B) ]`

each `normalize` = unit sum (so `λ` is meaningful and neither swamps the other;
alignment lives at the controller fusion point, **not** in the partition-agnostic
producers — §3). `λ=0` recovers A-only.

**Additive, not multiplicative — the load-bearing reason for B's existence.** B's
job is to surface error the *gradient proxy misses* (the §4.1 caveat). A
multiplicative `d_A·(1+α·d_B)` gates B by A: where the gradient is blind (`d_A≈0`)
B cannot raise demand — defeating the correction. Additive lets B independently
light up a cell A missed.

**Cost of B (must be reported).** Per B evaluation, `compute_gaussian_score_fastgs`
does **2 forward renders per camera** in its `camlist` (forward-only, no backprop):
one for the photometric loss, one to accumulate per-Gaussian high-error counts. So
cost ≈ `2·|camlist|` forward renders per evaluation. Two knobs bound it: the camera
subsample `|camlist|` and the **B-period `M`** (controller steps between B
evaluations; `d_B` is held between refreshes). Conservative defaults (small
subsample, periodic) keep it a few-percent overhead; full-camera every step would
obliterate the compute-saving by-product. *(Wall-clock not yet measured; the
A+B-vs-A training-time delta is reported in §6.3 Exp 4.)*

**Fallback (data-driven, not pre-committed):** if the ablation shows B's quality
gain does not justify its render cost, demote B to a **validation-only** diagnostic
(check that A's reallocation correlates with true photometric error) and ship
A-only.

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
- Recompute demand field + run controller every **N iters** — one *controller step*
  (aligned with Octree-GS `check_interval = 100`).
- **One shared smoothing time constant `τ_smooth`** (in controller steps) governs
  both the EMA on the demand field and the convergence gate (§5):
  - EMA: `β = 1 − 1/τ_smooth`, smoothing the demand field over ≈`τ_smooth` steps to
    damp grow/prune jumpiness.
  - The §5 Spearman gate compares `d_smooth` **`τ_smooth` steps apart** (see below).
- **Why one constant, not two.** The Spearman comparison horizon must be
  `≥ τ_smooth` (the EMA memory): comparing closer than the smoothing memory measures
  the filter's autocorrelation, not whether the signal re-ranked. Setting the
  horizon **equal** to `τ_smooth` is the tightest valid choice — so the two are not
  independent knobs. Default `τ_smooth = 3` (≈300 iter @ N=100); ablation axis.

### 4.5 Pluggable interface

```python
class DemandProducer:
    # Contract: return raw, non-negative, comparable per-ANCHOR Anchor Demand s(a)
    # in [0, +inf), one per anchor. No units, no normalization, no knowledge of
    # cells/control_level (partition-agnostic). The s(a) → d(v) reduction lives in
    # the Partition module; the single L1 normalization lives in the controller.
    def produce(self, scene, stats) -> Tensor:  # shape [N_anchors] = s(a)
        ...

# now:    ErrorVisibilityDemand (§4.1)  — reads training_statis (+ FastGS)
# future: SemanticDemand (semantic mask → per-anchor score, same contract)
```

The per-anchor `s(a)` is reduced to the per-cell **Demand Field** `d(v)` by the
**Partition** module (`d(v) = segment_sum(s(a), Cell Membership)`); see §3.

## 5. Budget Controller

A pure function with a hard conservation guarantee.

**Inputs:** Demand Field `d(v)` (gradient, EMA-smoothed + periodic photometric
correction), current Cell Occupancy `n(v)`, Capacity Budget `B_total`.

**Activation & lifecycle (progressive-training coupling).** The whole demand
controller activates **only after full progressive unlock**, for two independent
reasons grounded in the Octree-GS code:
1. *Blind demand field.* Anchors exist at all levels from init (`octree_sample`),
   but progressive `set_anchor_mask` masks levels finer than the current
   `coarse_index` out of rendering — masked anchors get no gradient/visibility, so
   the demand field is dark at fine granularity until unlock.
2. *Gated actuator (decisive).* The finer-spawning `ds` branch of `anchor_growing`
   (our capacity-deepening mechanism) is itself gated on
   `iteration > coarse_intervals[-1]` — so demand-driven finer growth physically
   cannot execute before full unlock. Hence the gate is **full unlock**, not merely
   "`control_level` unlocked".

- **Activation iteration** = `coarse_intervals[-1]` when `progressive` (default
  schedule: `coarse_iter = 10000` ⇒ full unlock at iter 10000), else `update_from`
  (1500). Before it, native Octree-GS coarse→fine training runs unmodified.
- **Reallocation window** = `(activation, update_until)` — with progressive on the
  defaults give `(10k, 25k) ≈ 15k iters` (`adjust_anchor` only runs before
  `update_until = 25000`). Phase 1 ramp + Phase 2 steady state must fit inside it
  (~150 controller steps at `N=100`); if a scene needs more, raise `update_until`.
- The Spearman gate is a natural **second layer**: at unlock the newly-lit fine
  anchors shift the ranking, so it will not pass until the full-tree demand field
  re-stabilizes — no premature Phase 2 at the unlock boundary.

**Step 0 — native-pruning coordination (accounted, not out-of-band).**
Octree-GS's `weed_out` is **not** a periodic pruner: it runs at init and as a
**candidate-birth filter inside `anchor_growing`** (rejects candidates whose level
no camera renders, by camera-geometry LOD coverage). It never removes established
anchors mid-training, so it cannot desync the plan — our grow count-cap simply
applies to the post-`weed_out` survivors. The only periodic prune of *established*
anchors is the opacity-based `prune_anchor` **inside `adjust_anchor`** — already at
the controller cadence, not an independent force. We fold it in as a
**demand-independent dead-anchor GC** that runs **first** each controller step; the
Controller then reads the **post-GC** `n(v)`. Rationale: a dead anchor (collapsed
opacity) in a *deficit* cell is never removed by the demand-prune (that cell is
growing, not pruning), so it would waste a slot — GC is orthogonal to demand and
must stay, but accounted. (GC only frees headroom that demand-driven growth reuses;
the three-part invariant is preserved.) Runs in both phases; no phase toggle.

**Step 1 — surplus/deficit** (on post-GC `n(v)`)

```
c*(v) = clamp(B_total · d(v)/Σd, floor, cap)   # target capacity
Δ(v)  = c*(v) − n(v)                            # >0 wants more, <0 has surplus
```

**Step 2 — translate to per-cell actuator parameters**

- **Grow (Δ>0) — count cap, not threshold tuning.** `anchor_growing` proposes
  candidate anchors (offset-gradient-driven spawning at grid cells) under the
  **global** threshold, unchanged. The Actuator then **caps each Control Cell to its
  `Δ(v)` highest-(proposing-)gradient candidates**, applied **before** the
  `cat_tensors_to_optimizer` materialization block (~`gaussian_model.py:791`) so
  there is no optimizer churn. The per-cell *count cap* — not threshold lowering —
  is what enforces exact counts and hence the Budget Constraint (threshold lowering
  only changes how aggressively candidates appear, never the exact number).
  Candidates rank by their **proposing offset gradient**, not `s(a)` (`s(a)` is
  undefined for not-yet-created anchors; it is the prune-side signal). **No
  force-fill:** a high-deficit cell with few candidates simply grows fewer
  (consistent with `Σn ≤ B_total`). A per-cell "threshold map" (lowering the
  threshold in deficit cells) is **unnecessary** given no force-fill — kept only as
  an optional ablation.
- **Prune (Δ<0):** in surplus cells, prune the `|Δ(v)|` lowest-`s(a)` anchors
  (reusing the demand signal for ranking; no new metric introduced).

*Implementation note:* `anchor_growing`/`prune_anchor` are **pure PyTorch** (the
only CUDA submodule is the rasterizer), so both the grow cap and the prune ranking
are pure-Python edits on the fork — isolatable and locally unit-testable.

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
  demand ranking has stabilized** — Spearman rank correlation of `d_smooth(t)` vs
  `d_smooth(t − τ_smooth)` (horizon = `τ_smooth`, §4.4), over their **shared**
  cells, `≥ 0.9`, **sustained for k (≈2–3) consecutive steps**. Two orthogonal
  knobs: `τ_smooth` sets the comparison horizon / smoothing scale, `k` sets how long
  stability must hold. Rank stability (not EMA magnitude stability) is the correct
  gate: pruning ranks by `s(a)`, and ranks can hold while magnitudes drift under
  global rescaling. *Plateau fallback:* if growth flattens below `B_total`, enter
  Phase 2 under the cap (see budget semantics below / §6.3).
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
- **Rate limit:** move at most `r%` of `B_total` per controller step (e.g. 5%) for
  smooth convergence (`r` distinct from the gate sustain count `k`). With the
  budget-aware ramp there is no overshoot cliff at the phase switch, so the rate
  limit only governs steady-state churn.
- EMA (§4.4) keeps the demand field itself from jumping.

**Setting `B_total`:**

- `B_total` is an **anchor** budget (the controller conserves anchors, not rendered
  Gaussians; rendered Gaussians = anchors × `n_offsets`, further opacity-masked, a
  floating derived quantity).
- Main experiments: `B_total = Octree-GS's final anchor count`, taken at training
  end where the count is **frozen** (`adjust_anchor` stops at `update_until = 25000`,
  so iter-25000 = final). Reproducible, unambiguous, metric-independent (not the
  PSNR-best checkpoint — that would couple `B_total` to the baseline's metric and
  break pure capacity-pairing). **Per-scene**, from a **fixed-seed** baseline run,
  the exact value reported (§7.3).
- Sweep `B_total` → quality-vs-#anchors Pareto curve (the money figure).

### 5.1 ReallocationPlan & controller test surface

**Plan type** (the Controller's pure output):
```
ReallocationPlan:
  delta:    Tensor[N_cells]   # int; >0 grow-up-to, <0 prune-count, 0 hold
  phase:    "ramp" | "steady" # tells the Actuator whether prune is allowed
  c_target: Tensor[N_cells]   # optional, for the capacity heatmap / debug
```
A single signed integer `delta` suffices (grow/prune are mutually exclusive per
cell). **Plan vs executed:** the Controller is pure and its *plan* satisfies the
invariants exactly; the Actuator may grow fewer than `δ⁺` (candidate-limited, no
force-fill, §5 grow), so executed occupancy `≤` planned `≤ B_total`. **Unit tests
assert plan properties**; the executed `≤` is an integration property.

**Constraint application order** (composition is where bugs hide — it is defined,
not incidental):
1. raw target `t(v) = B_total · d(v)/Σd`;
2. **water-fill** floor/cap: clamp to `[floor, cap]`, redistribute the residual
   `B_total − Σt` over unclamped cells proportionally, iterate to fixpoint;
3. **integer apportionment** (largest-remainder / Hamilton): floor the targets,
   give the remaining `R = B_total − Σ⌊·⌋` units to the largest fractional
   remainders ⇒ `Σ target = B_total` exactly and integer;
4. `delta = target − n(v)`;
5. **dead-band**: `|delta| < θ → 0`;
6. **rate-limit**: scale so `Σ|delta| ≤ r%·B_total` (proportional);
7. **steady re-balance**: dead-band/rate-limit can break `Σδ = 0`; trim the
   marginal grow/prune to restore `Σδ = 0` (steady). Ramp instead clamps `δ ≥ 0`.
   *(Without step 7 the "exact conservation" invariant fails whenever the dead-band
   fires — this step is load-bearing.)*

**Invariants (asserted by tests):** integer `delta`; `Σn + Σδ ≤ B_total`; steady
`Σδ = 0` (P-in = P-out), ramp `δ ≥ 0` and `Σδ = B_total − Σn`; `floor ≤ target ≤
cap` per active cell; **determinism** (identical input → identical plan; tie-breaks
in apportionment and `s(a)` ranking are deterministic); **rank-monotonicity**
(`target` non-decreasing in `d(v)`, modulo floor/cap); **fixed-point/no-thrash**
(stable demand over consecutive steps → `δ ≈ 0`).

**Test matrix** = nine single-call cases (uniform@budget, uniform-ramp, skewed,
cap, floor, rate-limit, dead-band, empty `N_active=0`, extreme `B_total∈{0,1}`)
**plus** the cases they miss: integer-apportionment exactness; **multiple
constraints binding at once** (order + step-7 re-balance); over/under-constrained
(`Σcap < B_total` undershoot; `Σfloor > B_total` must error, not silently — the
`control_level` derivation should already preclude it, §4.2); multi-step
no-thrash/fixed-point; determinism/tie-breaks; rank-monotonicity. Each case asserts
both the invariant and physical reasonableness (e.g. a high-demand cell receives
`≥` the mean allocation).

## 6. Evaluation Plan

### 6.1 Baselines

| Method | Role |
|--------|------|
| Vanilla 3DGS (gsplat) | capacity-agnostic reference |
| **Octree-GS** | **primary base / control** (= uniform allocation; ours = + demand reallocation) |
| CLoD-GS | continuous-LOD comparison |
| FastGS | training-speed axis (optional) |

The critical comparison is **vs Octree-GS at equal #anchors**, isolating the
demand-reallocation variable.

### 6.2 Metrics

- Quality: PSNR / SSIM / LPIPS.
- Budget: **#anchors (the controlled variable / budget)**.
- Derived/secondary (reported, not controlled): rendered #Gaussians (opacity-masked),
  memory, render FPS, training time (the "less compute" by-product).

### 6.3 Experiments

1. **Main comparison — two operating points, both points on the Pareto curve:**
   - **Matched-budget:** force equality `Σn ≡ B_total = Octree-GS final #anchors`
     (plateau off; floor fills the budget). Strictly equal #anchors → "same budget,
     higher quality"; answers the "your N must match" objection.
   - **Natural-budget:** the cap `Σn ≤ B_total` (plateau allowed). #anchors `≤`
     baseline at higher quality → "less budget, higher quality" (stronger claim;
     #anchors reported explicitly).
2. **Money figure — Pareto curve:** sweep `B_total ∈ {0.25, 0.5, 1, 2}× baseline`
   → quality. Claim: our curve dominates across budgets.
3. **Ablations:**
   - Demand source: uniform (= Octree-GS) / gradient-only (`λ=0`) / gradient+
     photometric (`λ ∈ {0.5, 1.0}`) / photometric-only. Plus B-cost knobs
     `|camlist|`, `M` (cost–quality trade-off; see §4.1 fusion).
     **Fairness condition:** hold the entire downstream pipeline identical across
     arms (`τ_smooth`, cadence, controller L1 normalization, floor/cap) and vary
     *only* the raw signal source — so any quality delta is attributable to the
     signal's informativeness, not to an incidental change of distribution shape.
     (No per-producer distribution alignment; that would erase the shape the
     ablation measures.)
   - Conservation: hard / soft / none.
   - Reallocation headroom: sweep **`τ`** (`control_level` is derived from
     `τ` + `B_total`, §4.2) — granularity vs slack trade-off.
   - Controller knobs: floor/cap, `τ_smooth` (shared smoothing/gate horizon), `k`
     (gate sustain count), rate-limit sensitivity.
4. **By-product:** training time / FPS (the compute-saving corollary). Includes the
   **A+B vs A-only training-time delta** — B's `2·|camlist|`-render cost must be
   shown, not hidden (decides the §4.1 fusion-vs-validation fallback).
5. **Qualitative:** per-cell Target Capacity heatmap showing capacity flowed to
   high-detail regions — also previews the future semantic version (swap heatmap
   for semantic ROI).

### 6.4 Datasets (Decision D6: standard + one large highlight)

**Selection principle:** the highlight scene is chosen to **maximize demand-field
non-uniformity** — the regime where reallocation has the most leverage. A uniform
demand field gives `c*(v) → B_total/N_active` (the controller degrades to uniform
allocation), so leverage grows with skew.

- **Standard: Mip-NeRF360, Tanks & Temples, Deep Blending** — the expected benchmark
  tables (main comparison + Pareto + ablations). These sit at the **near-uniform**
  end, so they double as the **graceful-degradation control**: we should match the
  baseline where there is nothing to reallocate (no harm).
- **Large-scale highlight: BungeeNeRF** (multi-scale, satellite→ground). Its extreme
  scale variation makes the demand field **highly non-uniform** (far = low demand,
  near = high detail), the strongest stage for the budget-reallocation story.
  Supported out of the box by the Octree-GS base (`train_bungeenerf.sh`).
- **MatrixCity dropped.** Aerial city capture is single-scale and geometrically
  regular ⇒ near-uniform demand ⇒ the method has no leverage; it is the wrong stage
  and invites the "why would reallocation help on a uniform scene?" objection. The
  standard benchmarks already cover the uniform end, so a second large uniform scene
  adds cost without evidence (YAGNI).

The narrative closes in one line: **the more skewed the demand, the more we win** —
near-uniform standard scenes ⇒ ≈ baseline (no harm); non-uniform BungeeNeRF ⇒ ≫
baseline.

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
- Fixed random seeds — including the seed for the baseline run that **defines each
  scene's `B_total`** (§5); the exact per-scene `B_total` value is reported.
- Octree-GS `arguments/` config system records every experiment's settings.

### 7.4 Repo layout (Decision D7: fork Octree-GS as body)

- `ocbgs/` — a fork of the Octree-GS codebase as the working baseline, with
  **minimal-intrusion** edits to `gaussian_model.py` / `train.py` to call the
  controller and apply the plan, plus three new isolated, CUDA-free modules:
  - `ocbgs/demand/` — `DemandProducer` interface + `ErrorVisibilityDemand`; emits
    per-anchor Anchor Demand `s(a)`. Partition-agnostic.
  - `ocbgs/partition/` — Cell Membership + `control_level` derivation + the pure
    `s(a) → d(v)` segment-sum reduction. Reused verbatim by future producers.
    **Membership is stateless**, not an incrementally-maintained map: each controller
    step it is recomputed as `cell_idx = floor((anchor_pos − init_pos) / cell_size)`
    — a vectorized floor-division (O(N_anchors), pure tensor, ms-scale at 10⁶), not a
    spatial-tree lookup. Statelessness is correct whether or not anchor positions
    move (it never holds stale state), strictly dominating an incremental `anchor→cell`
    dict (which breaks on boundary crossing). **Assumption (asserted in config):**
    `position_lr = 0` — Octree-GS freezes anchor positions (only `_offset` moves),
    so an anchor's accumulated `s(a)` over a controller window belongs to a single
    cell (no demand smearing across a boundary).
  - `ocbgs/controller/` — the pure-function `BudgetController` + plan types;
    cell-level only (decides *how many*).
  The Actuator lives as minimal-intrusion hooks in the forked `gaussian_model.py`
  (it must mutate anchor state and decides *which* to prune via `s(a)`).
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
