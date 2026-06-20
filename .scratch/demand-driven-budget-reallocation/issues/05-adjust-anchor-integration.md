# Issue: `adjust_anchor` integration — dual-path gating + execution reorder + grow-cap + single optimizer surgery

**Status:** ready-for-agent

## What to build

Rewrite the body of `gaussian_model.adjust_anchor` to wire the full closed-loop pipeline (DemandProducer → Partition → BudgetController → Actuator) into the training loop. The `train.py` call site is unchanged. This is the first issue that changes training behaviour: the Controller plan is executed, anchors are pruned and grown per demand.

**Construction & attachment.** During model init, construct one `ErrorVisibilityDemand`, one `Partition`, and one `BudgetController` instance and attach them as `self.demand_producer`, `self.partition`, `self.controller`. `self.B_total` is read from the training config (the per-scene baseline anchor count measured at `update_until`, spec §5) — it is NOT hardcoded. The `train.py` call site is unchanged — `adjust_anchor(iteration, ...)` keeps the same signature.

**Dual-path gating — `controller_active(iteration)`:**

```
def adjust_anchor(self, iteration, ...):
    if not self.controller_active(iteration):
        return self._native_adjust_anchor(...)   # byte-equivalent native path
    # Controller path below...
```

- `controller_active(iteration)` = True when:
  - **Progressive mode** (default): `iteration > coarse_intervals[-1]` AND `iteration ≤ update_until`.
  - **Non-progressive mode**: `iteration > update_from` (default 1500) AND `iteration ≤ update_until`.
   Two independent reasons for the progressive activation threshold per ADR-0004 § activation: (i) the demand field is blind at fine granularity until unlock; (ii) the finer-spawning `ds` branch of `anchor_growing` is itself gated on `iteration > coarse_intervals[-1]`. In non-progressive mode, `coarse_intervals` does not exist — the `update_from` threshold replaces it (spec §5). The gate also enforces `iteration ≤ update_until` for three principled reasons per ADR-0004 § lifecycle: (i) densification policy belongs to the structure-convergence phase only — the ideal terminal state is `δ → 0` handed off to native parameter fine-tuning; (ii) `B_total` is measured at `update_until` — extending only the controller would densify under a different schedule, breaking equal-#anchors isolation; (iii) an idle controller in the tail still pays demand evaluation cost for zero structural change.
- Pre-unlock: dispatch to `_native_adjust_anchor` — the original Octree-GS body, moved verbatim into a private method. The baseline is unaffected.

**Controller path execution order (ADR-0005 §2):**

```
Step 0: GC_mask = self._opacity_dead_mask(self.opacity_accum, self.min_opacity) → Tensor[bool]
Step 1: s_a = self.demand_producer.produce(self, self.training_statis) → Tensor[N]
Step 2: cids, n = self.partition.reduce(positions, ones, exclude=GC_mask)  → post-GC occupancy
        cids, d_a = self.partition.reduce(positions, s_a, exclude=GC_mask) → demand field (Source A)
Step 3: plan = self.controller.plan(cell_ids=cids, d_A=d_a, occupancy=n, B_total=self.B_total)
        → ReallocationPlan
Step 4: anchor_cell_ids = self.partition.cell_id(positions)          → Tensor[N_anchors]
Step 5: demand_prune_mask = self._lowest_sa_in_surplus(plan, s_a,
                                                       anchor_cell_ids) → Tensor[bool]
Step 6: self.prune_anchor(GC_mask | demand_prune_mask)               → ONE optimizer surgery
Step 7: self.anchor_growing_capped(plan, self.global_threshold)      → grow with count-cap
```

Key constraints:
- **Single `prune_anchor` call** (step 6) — `GC_mask ∪ demand_prune_mask` in one operation. The native separate prune call does NOT also fire (no double mutation).
- **Flipped order** (prune → grow vs. native grow → prune): free P slots first, then fill at most P slots. Controller reads post-GC `n(v)` analytically through Partition's `exclude` mask, not from post-mutation state.
- `set_control_level()` called once at the first post-unlock step, frozen thereafter.

**`anchor_growing_capped(plan, global_threshold)` — grow count-cap (ADR-0005 §5):**

- Calls native `anchor_growing` logic to propose candidates by the global gradient threshold (unchanged).
- Bins candidates by `Partition.cell_id()` (non-deficit Control Cells → empty set).
- In each deficit Control Cell (`δ(v) > 0`): keep top `δ⁺(v)` candidates ranked by their **proposing offset gradient**, discard the rest.
- The cap is inserted **before** `cat_tensors_to_optimizer` (~:791) — candidates are truncated before optimizer registration.
- Native accumulator padding (`offset_denom`, `offset_gradient_accum`, ~:863/:869) is computed from the capped count.
- **No force-fill:** a high-deficit Control Cell with few candidates simply grows fewer anchors — `executed ≤ planned ≤ B_total`.
- `weed_out` is untouched; it remains the candidate-birth filter inside `anchor_growing`. The cap applies to post-`weed_out` survivors.

**Summary of changes from native `adjust_anchor`:**

| Change | Detail |
|--------|--------|
| Added | `_opacity_dead_mask(opacity_accum, min_opacity)`, `_lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)`, `anchor_growing_capped()` |
| Reordered | prune → grow (flipped from native grow → prune) |
| Consolidated | single `prune_anchor` call (GC ∪ demand, not two calls) |
| Gated | `controller_active(iteration)` branches two paths |

## Acceptance criteria

### Automated

- [ ] `executed ≤ planned ≤ B_total`: after one Controller step on real anchor state (server), the total anchor count does not exceed `B_total`, and the executed count is ≤ the planned count. This is the integration property deferred from issue 03a — the Controller's plan satisfies its invariants exactly (unit-tested in 03a); this test verifies the *executed* side with real anchor state.

### Human diff review vs native Octree-GS `adjust_anchor`

Diff the rewritten `adjust_anchor` against the native version and confirm each:

- [ ] **1. Call-site unchanged** — `train.py` calls `adjust_anchor(...)` with the same signature and at the same location.
- [ ] **2. Native path byte-equivalent** — pre-unlock branch dispatches to `_native_adjust_anchor` with no behavioral drift; the baseline is unaffected. In progressive mode the gate is `iteration <= coarse_intervals[-1]`; in non-progressive mode the gate is `iteration <= update_from`.
- [ ] **3. Single optimizer surgery** — exactly ONE `prune_anchor(GC_mask | demand_prune_mask)` call in the controller branch; the native separate prune call does NOT also fire (no double mutation).
- [ ] **4. Execution order GC→plan→prune→grow** — flipped from native grow→prune; no residual native grow-before-prune path in the controller branch.
- [ ] **5. Optimizer-state rows stay in sync across prune & grow** — every per-anchor tensor and the Adam momentum states are sliced (prune) / padded (grow) consistently. Diff that ALL of these are handled: `_anchor`, `_offset`, `_scaling`, `_rotation`, `_anchor_feat`, `_opacity`; the accumulators `opacity_accum`, `anchor_demon`, `offset_gradient_accum`, `offset_denom`; and the per-param `exp_avg` / `exp_avg_sq` in every optimizer param group. A row added/removed from `_anchor` but not from `offset_denom` or the momentum buffers desyncs silently.
- [ ] **6. Grow-cap insertion point** — `anchor_growing_capped` truncates candidates BEFORE `cat_tensors_to_optimizer` (~:791), inside `anchor_growing` (not a wrapper); native accumulator padding (`offset_denom`, `offset_gradient_accum`, ~:863/:869) is computed from the capped count.
- [ ] **7. weed_out untouched** — still the candidate-birth filter inside `anchor_growing`; the cap applies to post-weed_out survivors.
- [ ] **8. `set_control_level` one-time guard** — derived once at the first post-unlock step and frozen; not per-step, not at construction.
- [ ] **9. `executed <= planned <= B_total`** — integration assertion present and green on a real server run.

## Blocked by

- 01-demand-producer-source-a (needs `ErrorVisibilityDemand` to attach to model)
- 02-partition-cell-membership (needs `Partition` for `reduce` and `cell_id`; needs `set_control_level` one-time guard)
- 03b-controller-temporal-layer (needs full `BudgetController` with phase logic)
- 04-actuator-pure-helpers (needs `_opacity_dead_mask` and `_lowest_sa_in_surplus`)
