# Issue: `ocbgs/controller/` — single-step static allocator (7-step constraint chain, internal `_allocate` with `phase` input)

**Status:** ready-for-agent

## What to build

Implement the BudgetController's **internal `_allocate(cell_ids, d, occupancy, B_total, phase)`** — a **stateless** pure function that produces a `ReallocationPlan` satisfying the three-part invariant. The `phase` parameter is a pure input (produced by issue 03b); this allocator implements **both** ramp and steady branches. This is a private method called by the public `plan()` (03b), not part of the public ABC.

The public `plan()` signature is `plan(cell_ids, d_A, occupancy, B_total, d_B=None) -> ReallocationPlan` (see issue 00). `phase` is determined internally by the temporal layer (03b); this issue only implements the stateless allocation mechanics.

**`ReallocationPlan` type:**

```
ReallocationPlan:
  cell_ids: Tensor[N_cells]   # stable Control-Cell ids from Partition
  delta:    Tensor[N_cells]   # int; >0 = grow-up-to, <0 = prune-count, 0 = hold
  phase:    "ramp" | "steady" # tells Actuator whether pruning is allowed
  c_target: Tensor[N_cells]   # optional, for capacity heatmap / debug
```

A single signed integer `delta` suffices — grow and prune are mutually exclusive per Control Cell.

**7-step constraint application order (ADR-0004 § composition):**

1. **L1 normalise raw target:** `t(v) = B_total · d(v) / Σd`
2. **Water-fill floor/cap:** clamp to `[floor, cap]`, redistribute the residual `B_total − Σt` over unclamped cells proportionally, iterate to fixpoint. `cap = min(k_cap · m, 0.25 · B_total)` where `m = B_total / N_active`. `floor` applies only to active cells.
3. **Integer apportionment:** Hamilton / largest-remainder — floor targets, give remaining `R = B_total − Σ⌊·⌋` units to cells with largest fractional remainders ⇒ `Σ target = B_total` exactly, all targets integer.
4. **Delta:** `δ(v) = target(v) − n(v)` (>0 = deficit, <0 = surplus)
5. **Dead-band:** `|δ(v)| < max(1, θ_frac · target(v)) → 0` (per-cell-relative, default `θ_frac = 0.25`)
6. **Rate-limit:** scale all deltas so `Σ|δ| ≤ r% · B_total` (global proportional, default `r% = 5%`)
7. **Steady re-balance (load-bearing):** In steady phase, dead-band and rate-limit can break `Σδ = 0`. Trim the marginal grow/prune entries to restore `Σδ = 0`. In ramp phase, instead clamp `δ ≥ 0` (no pruning).

**Controller defaults (fixed pipeline):**

| knob | symbol | default | unit |
|------|--------|---------|------|
| min mean occupancy | `ρ_min` | 8 | anchors/cell — (Partition knob, referenced) |
| floor | `floor` | 1 | anchors/cell (absolute min) |
| cap multiplier | `k_cap` | 8 | × mean occupancy |
| dead-band | `θ_frac` | 0.25 | fraction of target per cell |
| rate limit | `r%` | 5% | fraction of B_total |
| smoothing/gate horizon | `τ_smooth` | 3 | Controller steps (03b knob) |

**Three-part testable invariant (one test asserts all three):**

1. Phase-2 reallocation conserves exactly — `Σ δ⁺ = Σ |δ⁻|`
2. `Σ target(v) ≤ B_total` at all times
3. When binding (capacity fully demand-justified), Phase-2 total `Σ target(v) ≡ B_total`

## Acceptance criteria

- [ ] Three-part invariant: single test case asserting all three parts simultaneously — `Σδ = 0` in steady, `Σ target ≤ B_total` always, `Σ target ≡ B_total` when binding
- [ ] `uniform@budget`: all cells equal demand in steady phase → equal targets
- [ ] `uniform-ramp`: equal demand in ramp phase → proportional fill, `δ ≥ 0` for all cells
- [ ] `skewed`: high-demand cell gets > mean target, low-demand gets < mean (rank-monotonicity verified implicitly)
- [ ] `cap-binds`: high-demand cell clipped at cap, residual redistributed to unclamped cells
- [ ] `floor-binds`: low-demand cell held at floor; floor not applied to empty (inactive) cells
- [ ] `rate-limit-binds`: `Σ|δ|` capped at `r% · B_total`, proportional scaling across cells
- [ ] `dead-band-binds`: small `|δ|` zeroed, step-7 marginal trim restores `Σδ = 0` in steady
- [ ] `multiple constraints binding at once`: cap + rate-limit both active — composition order verified, step-7 re-balance holds
- [ ] `integer apportionment exactness`: Hamilton largest-remainder produces exactly `Σ target = B_total` with integer targets; tie-break behaviour is deterministic
- [ ] `Σfloor > B_total`: must raise an error (safety property — `control_level` derivation should preclude this, but the guard must exist)
- [ ] `empty N_active = 0`: does not crash, returns empty plan
- [ ] `ramp` branch: `p = (B_total − N_total) / ΣΔ⁺` proportional clamp when crossing budget threshold — delta scaled so total lands on `B_total` without overshoot
- [ ] All operations are tensor-level arithmetic (sum, clamp, sort via largest-remainder); no CUDA import in `ocbgs/controller/`

## Blocked by

- 00-walking-skeleton (needs `ReallocationPlan` type and `BudgetController` ABC — allocation is tested with synthetic `(d, n)` tensors, no real Partition required)
