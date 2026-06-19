# ADR-0005: Actuator integration & hook points

**Status:** Accepted
**Source spec:** `docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`

## Context

The BudgetController (ADR-0004) produces a ReallocationPlan — per-Control-Cell count instructions (how many to grow or prune). But the plan does not touch anchors, does not rank them by `s(a)`, and does not mutate optimizer state.

The Actuator is the only unit that mutates anchor state and the only unit that executes grow/prune surgery. It consumes the plan (how many) and decides **which** anchors grow or die, then performs the optimizer mutation. It is also the second consumer of `s(a)` (ADR-0001, constraint 1: the first is Partition's reduction `s(a) → d(v)`, ADR-0003; the Actuator uses `s(a)` for prune ranking).

The integration seam is `adjust_anchor`: its body is rewritten (not an external wrapper), but the `train.py` call site is unchanged. `gaussian_model.adjust_anchor` becomes a thin orchestrator + tensor-mutation seam; the pure logic lives in `ocbgs/partition/` (ADR-0003) and `ocbgs/controller/` (ADR-0004). All edits are pure PyTorch — the only CUDA submodule is the rasterizer, which is never modified (ADR-0001).

## Decision

### 1. Two activation paths

The Controller activates only after full progressive unlock (ADR-0004). `adjust_anchor` is gated by `controller_active(iteration)`:

```
def adjust_anchor(self, iteration, ...):
    if not self.controller_active(iteration):
        return self._native_adjust_anchor(...)

    s_a   = self.demand_producer.produce(self, self.training_statis)   # 0002 pluggable interface
    GC_mask = self._opacity_dead_mask()                                  # Step 0 dead-anchor GC

    positions = self.get_anchor_positions()
    cids, n = self.partition.reduce(positions,
                                    torch.ones_like(s_a),
                                    exclude=GC_mask)     # 0003: occupancy = unit-weight reduce
    _,    d = self.partition.reduce(positions,
                                    s_a,
                                    exclude=GC_mask)     # 0003: d(v)

    plan  = self.controller.plan(cell_ids=cids, d=d,    # 0004: plan keyed by cids
                                  occupancy=n, B_total=self.B_total)

    prune_set = GC_mask | self._lowest_sa_in_surplus(plan, s_a)
    self.prune_anchor(prune_set)                         # ONE optimizer surgery (GC ∪ demand)
    self.anchor_growing_capped(plan, self.global_threshold)  # count-cap at :791
```

- **Pre full-unlock** (`iteration ≤ coarse_intervals[-1]`, progressive): `_native_adjust_anchor` runs unmodified — establishes the multi-resolution skeleton before the Controller activates.
- **Post-unlock**: the Controller path above.
- `demand_producer`, `partition`, and `controller` are constructed once and attached to the model; `adjust_anchor` references them via `self`. The `train.py` call site is unchanged.

### 2. Execution order — GC → plan → prune → grow

The native Octree-GS order is grow → prune. We flip it to prune → grow.

Rationale: first free P slots (prune surplus + GC dead anchors), then fill at most P slots (grow deficit). The accumulator slice (prune) precedes the pad (grow). The Controller reads post-GC `n(v)` analytically through Partition's `exclude` mask (ADR-0003), not from post-mutation state — the Controller's plan is computed before any anchor is touched.

### 3. Step 0 — dead-anchor GC mask generation

This is the "how to detect dead anchors" decision that ADR-0004 defers here.

`_opacity_dead_mask()` identifies anchors with collapsed opacity (`opacity_accum` below threshold). Accounting rationale: GC is orthogonal to demand and must remain accounted (ADR-0004, Step 0).

A single `prune_anchor` call covers `GC_mask ∪ demand_prune_mask` — **one optimizer surgery**, not two separate calls. The Controller sees post-GC occupancy analytically (via Partition `exclude`), but the optimizer is mutated only once.

### 4. Prune-by-`s(a)` — the Actuator's sole consumption of `s(a)`

In each surplus Control Cell (`δ(v) < 0`), select the `|δ(v)|` anchors with the lowest Anchor Demand `s(a)` and add them to the prune set.

This is the Actuator's only use of `s(a)`, and the second fan-out consumer of `s(a)` in the architecture (ADR-0001, constraint 1). The first consumer is Partition's reduction `s(a) → d(v)` (ADR-0003).

**Tripwire — two ranking signals, never confused.** Grow ranks candidates by **proposing offset gradient** (see Decision 5); prune ranks established anchors by **`s(a)`**. `s(a)` is undefined for not-yet-created grow candidates — it is a prune-side signal. Confusing the two would break the grow cap's semantics.

### 5. Grow count-cap — not threshold tuning

`anchor_growing` proposes candidate anchors by the **global** gradient threshold, unchanged. The Actuator then **caps each Control Cell's candidates**: bin them by `Partition.cell_id()` (ADR-0003), keep the top `δ⁺(v)` candidates ranked by their **proposing offset gradient**, and discard the rest.

The cap is inserted **before** the `cat_tensors_to_optimizer` block (`gaussian_model.py:~791`) — candidates are truncated before optimizer registration, so there is no optimizer churn from adding then removing. The native accumulator padding (`offset_denom`, `offset_gradient_accum`, `:863/:869`) is computed from the resulting (capped) anchor count, so a smaller capped count automatically pads less — the cap integrates with native bookkeeping for free.

**Tripwire — this is a count-cap, not threshold tuning.** A per-cell count cap enforces exact counts and hence the Budget Constraint. Lowering the threshold only changes how aggressively candidates appear, never the exact number. A per-cell "threshold map" (lowering the threshold in deficit cells) is unnecessary given no force-fill, and is kept only as an optional ablation.

**No force-fill.** A high-deficit Control Cell with few candidates simply grows fewer anchors — consistent with `Σn ≤ B_total` (ADR-0004). The Controller plan's `δ⁺` is an upper bound, not a mandatory quota. Candidate-limited under-execution is expected behaviour, not a bug.

### 6. `weed_out` coordination

`weed_out` is not a periodic pruner. It runs only at init and as a **candidate-birth filter inside `anchor_growing`** — it rejects candidates whose octree level no camera renders (by camera-geometry LOD coverage). It never removes established anchors mid-training, so it cannot desync the ReallocationPlan.

The grow count-cap applies to the post-`weed_out` survivors — it caps who survives the birth filter. `weed_out` does not appear as a separate step in the `adjust_anchor` orchestration; it is an internal detail of `anchor_growing`.

### 7. `executed ≤ planned` — integration verification

This is the integration property deferred from ADR-0004. The Controller's plan satisfies `Σδ = 0` and `Σn ≤ B_total` exactly (unit tests assert plan properties). The Actuator's executed occupancy may be lower than planned due to candidate-limited cells (no force-fill). Therefore:

`executed occupancy ≤ planned occupancy ≤ B_total`

This is an **integration test** property — it requires real anchor state and candidate generation, not synthetic tensors. It is verified at the Actuator level, not the Controller level.

### 8. `adjust_anchor` rewrite — architecture-level constraints

- `train.py` call site is unchanged: `adjust_anchor(...)` keeps the same argument signature and call location.
- `demand_producer`, `partition`, and `controller` are constructed once and attached to the model.
- `gaussian_model.adjust_anchor` is a thin orchestrator — it contains no pure logic (that lives in ADR-0003 and ADR-0004), only the call orchestration and tensor mutation.
- The grow cap must be inserted **inside** `anchor_growing` — it cannot be a wrapper because candidates are materialised inline before `cat_tensors_to_optimizer`.
- Summary of changes from native `adjust_anchor`:
  - **Added:** `_opacity_dead_mask()`, `_lowest_sa_in_surplus()`, `anchor_growing_capped()`
  - **Reordered:** prune → grow (flipped from native grow → prune)
  - **Consolidated:** single `prune_anchor` call (GC ∪ demand, not two calls)
  - **Gated:** `controller_active(iteration)` branches two paths

## Consequences

- The Actuator is **pure PyTorch** — grow and prune are Python tensor operations. It requires anchor state for integration tests, but no CUDA rasterizer. Locally testable on Windows.
- `s(a)` is consumed here exclusively for prune ranking; grow ranking uses proposing offset gradient. The two signals are never interchangeable.
- No force-fill means the Controller plan's `δ⁺` is an upper bound, not a mandatory fill quota. Candidate-limited cells naturally grow fewer anchors — this preserves `Σn ≤ B_total` without corrective logic.
- `weed_out` requires no additional coordination — it is an internal `anchor_growing` filter, not a separate periodic operation.
- The `executed ≤ planned` integration property is verified with real anchor state at the Actuator level.

## Non-goals

- Computation of the ReallocationPlan (water-fill, integer apportionment, phase logic, three-part invariant) — ADR-0004. The Actuator only consumes the plan.
- Implementation of `Partition.cell_id()` and `Partition.reduce()` — ADR-0003. The Actuator only calls `cell_id` for candidate binning.
- Definition and production of `s(a)` — ADR-0002. The Actuator only consumes `s(a)` for prune ranking.
- Demand Field computation, A+B fusion, EMA smoothing — ADR-0002 and ADR-0004.
- Any modification to the CUDA rasterizer — ADR-0001. All Actuator edits are pure PyTorch.
- `B_total` measurement and Pareto sweep — ADR-0004; measurement protocol in `docs/eval-plan.md`.
