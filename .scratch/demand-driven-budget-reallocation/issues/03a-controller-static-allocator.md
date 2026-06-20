# Issue: `ocbgs/controller/` — single-step static allocator (7-step constraint chain, internal `_allocate` with `phase` input)

**Status:** DONE

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

`N_active` = count of **active** Control Cells, where active = `occupancy(v) > 0 OR d(v) > 0`. Inactive cells (empty space) get neither `floor` nor a share of the mean `m`; they are excluded from `N_active`, from `m = B_total / N_active`, and from the water-fill set.

1. **L1 normalise raw target:** `t(v) = B_total · d(v) / Σd`. Degenerate `Σd = 0` → uniform `t(v) = B_total / N_active` over active cells.
2. **Water-fill floor/cap:** clamp to `[floor, cap]`, redistribute the residual `B_total − Σt` over unclamped active Control Cells proportionally, iterate to fixpoint (with an iteration cap + tolerance to guarantee termination on floats). `cap = min(k_cap · m, 0.25 · B_total)` where `m = B_total / N_active`. `floor` applies only to active Control Cells. When every active cell is capped (`Σcap < B_total`, the **undershoot** regime), water-fill terminates with `Σt = Σcap < B_total` — there is no unclamped cell to absorb the residual.
3. **Integer apportionment:** Hamilton / largest-remainder over the **post-water-fill** target sum. Let `t_sum = Σ clamped_t`. Floor each target, then give the remaining `R = round(t_sum) − Σ⌊·⌋` units to the Control Cells with the largest fractional remainders ⇒ `Σ c* = round(t_sum)`, all `c*(v)` integer. **`Σ c* = round(t_sum) ≤ B_total` always — equal to `B_total` only when binding (no collective cap undershoot); strictly less in the undershoot regime.** The apportionment pool is `round(t_sum)`, **not** `B_total`.
4. **Delta:** `δ(v) = c*(v) − n(v)` (>0 = deficit, <0 = surplus)
5. **Dead-band:** `|δ(v)| < max(1, θ_frac · c*(v)) → 0` (per-Control-Cell-relative, default `θ_frac = 0.25`)
6. **Rate-limit (steady only — skipped in ramp, see F4 below):** scale all deltas so `Σ|δ| ≤ r% · B_total` (global proportional, default `r% = 5%`). `scale = min(1, (r% · B_total) / Σ|δ|)` — the `min(1, ·)` guard is load-bearing: without it, when `Σ|δ| ≤ r% · B_total` (not binding) the ratio exceeds 1 and would **inflate** the deltas. Re-integerise with `trunc` (round toward zero) so the bound `Σ|δ| ≤ r% · B_total` still holds after rounding; trunc may break `Σδ = 0`, which is exactly why step 7a follows.
7. **Phase branch:**
   - **7a — Steady re-balance (load-bearing):** dead-band and rate-limit can break `Σδ = 0`. Restore it by **unit-level marginal trim on the surplus side only**: let `net = Σδ`. If `net > 0`, stable-sort the positive `δ` ascending and absorb `net` units via prefix-sum — fully zero the smallest grow entries, then partially reduce the single marginal boundary entry by the leftover. If `net < 0`, symmetric on the negative `δ` (increment toward zero). `net ≤ Σδ⁺` (and `|net| ≤ Σ|δ⁻|`) always holds, so the surplus side can always absorb it. **Do not** zero whole entries blindly and **do not** touch the wrong sign — both break `Σδ = 0`.
   - **7b — Ramp clamp:** clamp `δ ≥ 0` (no pruning). If `N_total + Σδ > B_total`, scale grow deltas by `p = (B_total − N_total) / Σδ⁺`, then re-integerise via **largest-remainder** so the total lands **exactly** on `B_total`: distribute the leftover `(B_total − N_total) − Σ⌊p·δ⌋` units to the largest fractional remainders. Plain `trunc` here would undershoot `B_total` by the sum of truncations.

**F4 — rate-limit is steady-only (decided: Option 1, skip in ramp).** ADR-0004 §stability ("rate limit only governs steady-state churn") and §phase-1 (ramp is governed by the budget-aware proportional clamp `p`) jointly fix this: ramp does **not** run step 6. Rationale: (1) ramp `δ⁺` is a *plan* upper bound — actual per-step growth is throttled by the Actuator's finite candidate supply (`executed ≤ planned`, ADR-0004 plan-vs-executed / ADR-0005), which is ramp's natural rate limiter; (2) the `p` clamp already pins `N_total + Σδ ≤ B_total` (no overshoot; `p → 0` freezes structure at budget); (3) applying rate-limit in ramp would pre-shrink `Σδ⁺`, architecting away `p` (which rarely fires once `Σδ⁺ ≤ 5% · B_total`) and stretching ramp to ≥20 steps — risky under the tight Scenario-B reallocation window (ADR-0004 §lifecycle). Per-step smoothing in ramp comes instead from the demand-field EMA (`τ_smooth`, issue 03b) and the candidate-supply ceiling.

**Controller defaults (fixed pipeline):**

| knob | symbol | default | unit |
|------|--------|---------|------|
| min mean occupancy | `ρ_min` | 8 | anchors/Control Cell — (Partition knob, referenced) |
| floor | `floor` | 1 | anchors/Control Cell (absolute min) |
| cap multiplier | `k_cap` | 8 | × mean occupancy |
| dead-band | `θ_frac` | 0.25 | fraction of `c*(v)` per Control Cell |
| rate limit | `r%` | 5% | fraction of B_total |
| smoothing/gate horizon | `τ_smooth` | 3 | Controller steps (03b knob) |

**Three-part testable invariant (one test asserts all three):**

1. Phase-2 reallocation conserves exactly — `Σ δ⁺ = Σ |δ⁻|`
2. `Σ c*(v) ≤ B_total` at all times
3. When binding (capacity fully demand-justified), Phase-2 total `Σ c*(v) ≡ B_total`

Note: these invariants assert *plan* properties (the Controller is a pure function). The integration property `executed occupancy ≤ planned occupancy ≤ B_total` is verified in issue 05 with real anchor state.

## Acceptance criteria

- [x] Three-part invariant: single test case asserting all three parts simultaneously — `Σδ = 0` in steady, `Σ c* ≤ B_total` always, `Σ c* ≡ B_total` when binding
- [x] `uniform@budget`: all Control Cells equal demand in steady phase → equal Target Capacity `c*(v)`
- [x] `uniform-ramp`: equal demand in ramp phase → proportional fill, `δ ≥ 0` for all Control Cells
- [x] `skewed`: high-demand Control Cell gets `>` mean Target Capacity, low-demand gets `<` mean (rank-monotonicity verified implicitly)
- [x] `cap-binds`: high-demand Control Cell clipped at cap, residual redistributed to unclamped Control Cells
- [x] `cap-undershoot` (Σcap < B_total): collective cap binding prevents full allocation — `Σ c* = round(Σcap) < B_total`, does not crash. This is the **only** case where part-2 of the invariant (`Σ c* ≤ B_total`) is strict rather than equality, so without it the `≤` assertion in the three-part invariant test is vacuous. Reachable at small `N_active` (e.g. `N_active = 2`, equal demand ⇒ each cap = `0.25·B_total` ⇒ `Σ c* ≈ 0.5·B_total`).
- [x] `floor-binds`: low-demand Control Cell held at floor; floor not applied to empty (inactive) Control Cells
- [x] `rate-limit-binds`: `Σ|δ|` capped at `r% · B_total`, proportional scaling across Control Cells
- [x] `dead-band-binds`: small `|δ|` zeroed, step-7 marginal trim restores `Σδ = 0` in steady
- [x] `multiple constraints binding at once`: cap + rate-limit both active — composition order verified, step-7 re-balance holds
- [x] `integer apportionment exactness`: Hamilton largest-remainder produces exactly `Σ c* = B_total` with integer `c*(v)`; tie-break behaviour is deterministic
- [x] `Σfloor > B_total`: must raise an error (safety property — `control_level` derivation should preclude this, but the guard must exist)
- [x] `empty N_active = 0`: does not crash, returns empty plan
- [x] `ramp` branch: `p = (B_total − N_total) / ΣΔ⁺` proportional clamp when crossing budget threshold — delta scaled so total lands **exactly** on `B_total` without overshoot, via largest-remainder re-integerisation of `p·δ` (plain `trunc` undershoots by the sum of truncations); rate-limit (step 6) is **not** applied in ramp
- [x] All operations are tensor-level arithmetic (sum, clamp, sort via largest-remainder); no CUDA import in `ocbgs/controller/`

## Blocked by

- 00-walking-skeleton (needs `ReallocationPlan` type and `BudgetController` ABC — allocation is tested with synthetic `(d, n)` tensors, no real Partition required)

## Review notes (non-blocking, can be addressed with 03b)

1. **B_total=0 steady semantics**: early return at `B_total <= 0` produces an all-zero plan (`c_target=0, delta=0`) — does NOT prune when `occupancy > 0` and `B_total=0`. This is a degenerate no-op guard, not a semantically correct shrinkage-to-zero. Add a docstring note.

2. **N3 floor-cap feasibility**: the true safety constraint is `0.25·B_total >= floor`, which holds for any realistic `B_total` (thousands). The `Σfloor > B_total` guard covers the opposite direction. The water-fill argument in N3 should reference `0.25·B_total`, not `B_total/N_active`.

3. **Rate-limit proportional scaling** (resolved): the core difficulty in testing step 6's per-cell scaling is that naive constructions produce post-trunc deltas with `net ≠ 0`, which step 7a then zeroes/collapses — erasing the scaling ratios. The fix uses **symmetric deltas** so `net=0` after trunc: `delta=[+40,+40,-40,-40]` at `B_total=320`, `scale=0.1`, `trunc→[4,4,-4,-4]`, `Σδ=0` → step 7a is a no-op and the ratio `4:4:4:4` is asserted directly via `torch.equal`. See `test_proportional_scaling_preserved_when_sigma_delta_zero`.

4. **Hamilton R<0 dead branch**: the `else: c_target = floor_t` branch (R <= 0) is unreachable — `round(t_sum) >= Σ⌊t⌋` always holds, and the `Σfloor > B_total` guard prevents over-allocation. Add an `# unreachable` comment.

5. **Step 7b `valid` mask**: the initial implementation used `valid = delta > 0` (post-floor mask), which incorrectly excluded cells whose `p·δ < 1` after floor. Fixed to `valid = delta_f > 0` (pre-floor mask). The new test `test_headroom_less_than_grow_cells_lands_exactly` (B_total=11, 10 cells, headroom=1) exercises this path — all `p·δ ≈ 0.111`, floor → 0, remaining=1 allocated via largest remainder.
