# ADR-0004: Budget Controller — conservation & phases

**Status:** Accepted
**Source spec:** `docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`

## Context

The Demand Field `d(v)` (ADR-0003) says where detail is needed, but not how much to allocate. The Budget Controller translates relative demand into exact per-Control-Cell counts under a hard Capacity Budget constraint. It is a pure function: inputs are the Demand Field, Cell Occupancy, and Capacity Budget; the output is a ReallocationPlan. It never touches `s(a)` (ADR-0001, constraint 2) and never mutates anchor state (that is the Actuator's job, ADR-0005).

The core tension: demand is relative and rank-meaningful; budget is an absolute integer. The Controller's job is a lossless translation between these two regimes — preserving the demand's shape (inter-cell ratios) while satisfying integer count constraints and the Budget Constraint.

## Decision

### D2 — Conservation target: anchors

`B_total` is an anchor budget. The Controller conserves anchors; rendered Gaussians (= anchors × `n_offsets`, further opacity-masked) and memory are floating derived quantities reported as by-products, not as the controlled variable.

### D5 — Hard upper-bound, two emergent phases

`Σ n(v) ≤ B_total` (an upper bound, not strict equality). The phase boundary is emergent from system state — no tuned `T_budget` iteration count. This mirrors how `control_level` is derived rather than chosen directly (ADR-0003).

### A+B fusion (receives the forward-reference from ADR-0002)

Source A (gradient-based) and Source B (FastGS photometric) are independently reduced to `d_A(v)` and `d_B(v)` by the Partition (ADR-0003). The Controller fuses them:

```
d(v) = EMA_τsmooth[ normalize(d_A) + λ · normalize(d_B) ]
```

Each `normalize` = unit sum (L1). The fusion operator is:

- **Additive** — not multiplicative. Multiplicative fusion `d_A · (1 + α · d_B)` gates B by A: where the gradient proxy is blind (`d_A ≈ 0`), B cannot raise demand, defeating the correction B exists to provide (ADR-0002). Additive fusion lets B independently light up a Control Cell A missed.
- **L1-normalised per signal** — so `λ` is a meaningful relative weight and neither signal swamps the other. Normalisation lives at the Controller, not in the partition-agnostic producers (ADR-0002).
- **EMA-smoothed** — with time constant `τ_smooth` (see Cadence & smoothing below).

`λ = 0` recovers A-only. `λ` is an ablation axis.

### Budget normalisation

`c*(v) = clamp(B_total · d(v) / Σd, floor, cap)`

- **L1 normalisation** `d(v) / Σd` — pure positive scaling. Removes absolute scale but preserves rank and all inter-cell ratios — i.e. preserves the demand's *shape*, which is the signal's information. The Demand Score contract (ADR-0002) need only be non-negative and cross-cell-comparable; units never reach the Controller.

- **floor** — a baseline Target Capacity that protects existing content from being starved. Applies only to active Control Cells (occupied or `d(v) > 0`); empty space gets no floor, so `Σ floor = floor · N_active`. Default `floor = 1` (one anchor — the physical minimum to represent anything; `0` lets a low-demand cell empty completely and punch a hole when a held-out view looks at it). The `ρ_min` guard (ADR-0003) keeps floor safe: with `m ≥ ρ_min ≫ floor`, floor eats only `floor / m ≈ 1 / ρ_min ≈ 12%` of the Capacity Budget.

- **cap** — prevents a single Control Cell from monopolising the budget. Set relative to mean occupancy: `cap = min(k_cap · m, 0.25 · B_total)` where `m = B_total / N_active`. The `k_cap · m` term scales the ceiling with granularity; the `0.25 · B_total` term is a hard monopoly guard. Default `k_cap = 8` (a high-demand Control Cell may hold up to 8× the mean). Deliberately not tight (`2×`): BungeeNeRF, the highlight scene, is built for extreme demand skew, so a tight cap would clip the very win it is meant to show. Ablate `k_cap ∈ {4, 8, 16}`.

### Constraint application order

The composition order is defined, not incidental — composition is where bugs hide. Seven steps:

1. **raw target** `t(v) = B_total · d(v) / Σd`
2. **water-fill** floor/cap: clamp to `[floor, cap]`, redistribute the residual `B_total − Σt` over unclamped cells proportionally, iterate to fixpoint
3. **integer apportionment** (Hamilton / largest-remainder): floor the targets, give the remaining `R = B_total − Σ⌊·⌋` units to the largest fractional remainders ⇒ `Σ target = B_total` exactly and all targets integer
4. **delta** `δ(v) = target(v) − n(v)` (>0 = deficit, <0 = surplus)
5. **dead-band** `|δ(v)| < max(1, θ_frac · target(v)) → 0` — per-cell-relative, a fraction of the Control Cell's own Target Capacity, not an absolute count or a `B_total` fraction. An absolute / `B_total`-scaled threshold would freeze every Control Cell: per-cell `δ` are `O(m)` (single digits at a healthy `m ≈ 8`), so any `B_total`-scaled threshold (e.g. `0.001 · B_total ≈ 100`) would always be larger than `|δ|`. Default `θ_frac = 0.25`.
6. **rate-limit** scale so `Σ|δ| ≤ r% · B_total` (proportional). A global `B_total` fraction — correctly global since it bounds a sum over cells. Default `r% = 5%`.
7. **steady re-balance** — dead-band and rate-limit can break `Σδ = 0`. Trim the marginal grow or prune entries to restore `Σδ = 0` in the steady phase. **This step is load-bearing:** without it, the "exact conservation" invariant fails whenever the dead-band fires. In the ramp phase, instead clamp `δ ≥ 0`.

### Three-part testable invariant

One unit test asserts all three simultaneously:

1. Phase-2 reallocation conserves exactly — `P_in = P_out`, no leak.
2. The total anchor count satisfies `Σ n(v) ≤ B_total` at all times.
3. When binding (capacity fully demand-justified), Phase-2 total `Σ n(v) ≡ B_total`.

### Two emergent phases (no tuned `T_budget`)

The phase switch is state-triggered, not a magic iteration count. A fixed count mis-fires both ways: too early prunes immature anchors, too late forces a huge overshoot prune.

**Phase 1 — demand-guided ramp.** Total anchor count grows toward `B_total`, guided by demand, with no forced pruning. As `N_total` nears `B_total`, growth is budget-aware: the crossing step's grow quota is scaled by a single global factor `p = (B_total − N_total) / Σ Δ⁺` so the total lands exactly on `B_total` instead of overshooting (proportional clamp — fair, no sampling bias while the demand signal may still be immature). Once at `B_total`, `p → 0` naturally freezes structure so anchors keep training to maturity until the gate opens.

**Phase switch (emergent).** Enter Phase 2 when `N_total ≥ B_total` AND the demand ranking has stabilised — Spearman rank correlation of `d_smooth(t)` vs `d_smooth(t − τ_smooth)` (horizon = `τ_smooth`), over their **shared** Control Cells (requires 0003's stable `cell_ids`), `≥ 0.9`, sustained for `k` (≈ 2–3) consecutive steps. Two orthogonal knobs: `τ_smooth` sets the comparison horizon / smoothing scale; `k` sets how long stability must hold.

Rank stability (not EMA magnitude stability) is the correct gate: pruning ranks by `s(a)`, and ranks can hold while magnitudes drift under global rescaling. A natural robustness property: at full progressive unlock, newly-lit fine anchors shift the ranking, so the Spearman gate will not pass until the full-tree demand field re-stabilises — no premature Phase 2 at the unlock boundary.

**Plateau fallback.** If growth flattens below `B_total`, enter Phase 2 under the cap — the scene cannot productively use the full budget, and the slack is left honest rather than padded with low-value anchors that would only hurt reported FPS/memory.

**Phase 2 — steady-state reallocation.** Prune surplus Control Cells to free P slots, then redistribute exactly P slots to deficit Control Cells proportional to `Δ⁺`.

### Stability (anti-thrash)

- **Dead-band** `θ_frac` — per-cell-relative, `max(1, θ_frac · c*(v))`. Default `0.25`.
- **Rate-limit** `r%` — global `B_total` fraction bounding a sum. Default `5%`. Distinct from the gate sustain count `k`. With the budget-aware ramp there is no overshoot cliff at the phase switch, so the rate limit only governs steady-state churn.
- **EMA** (`τ_smooth`, see below) keeps the demand field itself from jumping.

### Cadence & smoothing

- Demand field + Controller run every `N` iterations (one Controller step), with cadence `N = update_interval = 100` (the `train.py` gate `iter % update_interval == 0`). The `check_interval` (also 100) is the maturity window inside `adjust_anchor` — same value, different role.
- **One shared smoothing time constant `τ_smooth`** (in Controller steps) governs both the EMA on the demand field and the Spearman comparison horizon:
  - EMA: `β = 1 − 1/τ_smooth`, smoothing the demand field over ≈`τ_smooth` steps.
  - The Spearman gate compares `d_smooth` `τ_smooth` steps apart.
- **Why one constant, not two.** The Spearman comparison horizon must be `≥ τ_smooth` (the EMA memory): comparing closer than the smoothing memory measures the filter's autocorrelation, not whether the signal re-ranked. Setting the horizon **equal** to `τ_smooth` is the tightest valid choice — so the two are not independent knobs. Default `τ_smooth = 3` (≈300 iterations at `N=100`); ablation axis.

### Controller defaults

The fixed pipeline for all ablations (vary one axis at a time). The load-bearing convention: every per-cell knob is expressed in units that scale with mean occupancy `m = B_total / N_active` (`cap`, `θ_frac`), so the Controller behaves identically across `control_level` choices; `floor` is the one absolute per-cell knob (the physical minimum, kept safe by `ρ_min`); `r%` is the one global knob (a `B_total` fraction bounding a sum).

| knob | symbol | default | units / form | ablate |
|---|---|---|---|---|
| min mean occupancy | `ρ_min` | 8 | anchors/cell — derives `control_level` (ADR-0003) | {4, 8, 16} |
| min active cells | `A_min` | 10 | cells | — |
| floor | `floor` | 1 | anchors/cell (absolute physical min) | — |
| cap | `k_cap` | 8 | `min(k_cap·m, 0.25·B_total)` | {4, 8, 16} |
| dead-band | `θ_frac` | 0.25 | `max(1, θ_frac·c*(v))` (per-cell relative) | — |
| rate limit | `r%` | 5% | `Σ\|δ\| ≤ r%·B_total` (global) | {2.5, 5, 10}% |
| smoothing/gate horizon | `τ_smooth` | 3 | Controller steps | sweep |
| gate sustain | `k` | 2–3 | consecutive steps | — |

These are initial first-principles values (scaling arguments), not yet empirically tuned; the ablations validate or refine them.

### Activation & lifecycle

**Activation.** The Controller activates only after full progressive unlock, for two independent reasons:

1. *Blind demand field.* Anchors exist at all levels from init, but progressive `set_anchor_mask` masks levels finer than the current `coarse_index` out of rendering — masked anchors get no gradient/visibility, so the demand field is dark at fine granularity until unlock.
2. *Gated actuator (decisive).* The finer-spawning `ds` branch of `anchor_growing` (the capacity-deepening mechanism) is itself gated on `iteration > coarse_intervals[-1]` — so demand-driven finer growth physically cannot execute before full unlock.

**Reallocation window** = `(activation, update_until)` — the Controller runs over exactly Octree-GS's native densification window. The `(update_until, iterations)` tail is native parameter fine-tuning on a frozen architecture for both our method and the baseline (no grow/prune, no demand stats, no Controller) — the structure/parameter separation is native to Octree-GS, not added by us.

**Decision — Controller stops at `update_until`, not `iterations`.** Three principled reasons:

1. *Densification policy belongs to the structure-convergence phase only.* The ideal terminal state is `δ → 0` (structure settled at `B_total`) handed off to native parameter fine-tuning.
2. *`B_total` is measured at `update_until`.* Extending only our Controller to `iterations` densifies under a different schedule than the count was measured under, breaking the equal-#anchors isolation. `update_until` is therefore a **single shared knob** that moves the Controller window and the baseline densification window together (see `B_total` definition below).
3. *An idle Controller in the tail still pays cost for zero benefit.* Even with `δ ≈ 0`, the demand evaluation cost fires every `M` steps — including FastGS's `2·|camlist|` forward renders (ADR-0002) — directly eroding the compute-saving by-product.

**Scenario-B guardrail.** Do not assume 150 steps always suffices. `B_total` = the baseline's final count, and the baseline uses roughly the whole window to reach it under the shared candidate-supply rate (we cap, never force-fill — see Plan vs Executed below), so the ramp alone can consume most of the window, leaving little room for Phase 2. Mitigation is per-scene, not a global extension: log whether Phase 2 was reached by `update_until`; if a scene systematically fails, raise `update_until` **for that scene and its baseline together** (re-measuring `B_total`), preserving symmetry.

### B_total — definition & relationship to baseline

`B_total` is the Capacity Budget the Controller conserves. Conceptually, it equals the baseline's final anchor count at `update_until` — the count is frozen at that point (`adjust_anchor` stops at `update_until = 25000`). This coupling is the reason the Controller window also ends at `update_until` (see lifecycle decision above): a window mismatch would densify under different schedules, breaking equal-#anchors isolation. The coupling is architectural, not an implementation convenience.

The measurement procedure (fixed-seed baseline run, exact per-scene count, reporting) and the Pareto sweep (`B_total ∈ {0.25, 0.5, 1, 2}× baseline → quality-vs-#anchors curve`) belong to the evaluation plan (`docs/eval-plan.md`).

### Step 0 — dead-anchor GC coordination

Octree-GS's only periodic prune of established anchors is the opacity-based `prune_anchor` inside `adjust_anchor`. We fold it in as a demand-independent dead-anchor GC that runs **first** each Controller step. The Controller then reads the **post-GC** `n(v)` (via Partition's `exclude` mask — the GC mask itself is generated by the Actuator, ADR-0005; the Controller only consumes the post-GC occupancy).

Rationale: a dead anchor (collapsed opacity) in a deficit cell is never removed by the demand-prune (that cell is growing, not pruning), so it would permanently waste a slot. GC is orthogonal to demand and must stay, but is accounted in the post-GC occupancy. GC only frees headroom that demand-driven growth reuses; the three-part invariant is preserved. Runs in both phases; no phase toggle.

### Plan vs Executed — the Controller only decides *how many*

The Controller is a pure function. Its **plan** satisfies the invariants exactly.

The Actuator (ADR-0005) may grow fewer than `δ⁺` — candidate supply is limited and there is no force-fill. Therefore executed occupancy `≤` planned `≤ B_total`.

**Unit tests assert plan properties** (invariant, determinism, rank-monotonicity, fixed-point). The `executed ≤ planned` inequality is an integration property verified at the Actuator level (ADR-0005). This boundary is critical: blurring it puts Actuator-side execution variance into the Controller's test surface.

### ReallocationPlan type

```
ReallocationPlan:
  cell_ids: Tensor[N_cells]   # stable Control-Cell ids from Partition (ADR-0003)
  delta:    Tensor[N_cells]   # int; >0 grow-up-to, <0 prune-count, 0 hold
  phase:    "ramp" | "steady" # tells Actuator whether prune is allowed
  c_target: Tensor[N_cells]   # optional, for capacity heatmap / debug
```

A single signed integer `delta` suffices — grow and prune are mutually exclusive per Control Cell. The plan is keyed by stable `cell_ids` so the Actuator can look up `δ(v)` for each cell by id (ADR-0003).

### Test surface

**Core invariant test** — a single test case asserting all three parts of the invariant simultaneously.

**Nine case-driven tests:**

| case | what it exercises |
|---|---|
| uniform@budget | all Control Cells equal demand, steady phase → equal targets |
| uniform-ramp | equal demand, ramp phase → proportional fill |
| skewed | high-demand Control Cell gets `>` mean, low-demand gets `<` mean |
| cap-binds | high-demand Control Cell clipped at `cap`, residual redistributed |
| floor-binds | low-demand Control Cell held at `floor` |
| rate-limit-binds | `Σ\|δ\|` capped at `r%·B_total`, proportional scaling |
| dead-band-binds | small `\|δ\|` zeroed, marginal trim restores `Σδ=0` |
| empty `N_active=0` | edge case: no active Control Cells |
| extreme `B_total ∈ {0,1}` | edge case: degenerate budget |

**Cases these nine miss (supplementary):**

- **Integer-apportionment exactness** — Hamilton / largest-remainder produces exactly `Σ target = B_total` with integer targets.
- **Multiple constraints binding at once** — e.g. cap + rate-limit both active. The composition order and step-7 re-balance are tested under combined constraint pressure. This is where composition bugs hide.
- **Over/under-constrained** — `Σcap < B_total` (undershoot: budget cannot be fully allocated; must not crash) and `Σfloor > B_total` (must error — the `control_level` derivation should already preclude it via the safety property, ADR-0003).
- **Multi-step no-thrash / fixed-point** — stable demand over consecutive steps produces `δ ≈ 0`.
- **Determinism / tie-breaks** — identical inputs produce identical plans.
- **Rank-monotonicity** — `target(v)` is non-decreasing in `d(v)`, modulo floor and cap.

Each case asserts both the invariant and physical reasonableness (e.g. a high-demand cell receives at least the mean allocation).

## Consequences

- The Controller is a pure function + CUDA-free: all operations are tensor-level arithmetic (sum, clamp, sort via largest-remainder), locally unit-testable on Windows.
- Cadence `N = 100` (aligned with Octree-GS `update_interval`), EMA smoothing over `τ_smooth = 3` Controller steps.
- The Phase switch depends on ADR-0003's stable `cell_ids` — the Spearman correlation is computed over cells shared across two steps.
- All per-cell knobs scale with mean occupancy `m`, so Controller behaviour is identical across `control_level` choices. Changing `ρ_min` (ADR-0003) changes the number of Control Cells but keeps per-cell dynamics consistent.
- Ablation axes: `λ` (A+B fusion weight, ADR-0002), `ρ_min` (ADR-0003), `k_cap`, `θ_frac`, `r%`, `τ_smooth`, `k`.

## Non-goals

- Any operation on `s(a)`. The Controller works purely at Control-Cell level on `d(v)` (ADR-0001, constraint 2).
- Production of `d_A(v)` and `d_B(v)`. These are the Partition's `reduce()` output (ADR-0003).
- Generation of the dead-anchor GC mask (opacity-dead detection). The mask is produced by the Actuator (ADR-0005); the Controller only reads post-GC `n(v)` via the `exclude` parameter.
- Grow count-cap materialisation, prune-by-`s(a)` execution, prune-then-grow order, `adjust_anchor` body rewrite — all Actuator territory (ADR-0005).
- `B_total` measurement procedure and Pareto sweep — evaluation plan (`docs/eval-plan.md`). The definition and architectural coupling to `update_until` are retained here.
