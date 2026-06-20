# Issue: `ocbgs/controller/` — temporal/phase layer (EMA smoothing + Spearman gate + phase-flag decision + plateau fallback)

**Status:** ready-for-agent

## What to build

Extend the BudgetController with the **stateful temporal layer** that owns EMA smoothing of the demand field, Spearman rank-stability gating, and phase-flag decision logic. This layer implements the public `plan(cell_ids, d_A, occupancy, B_total, d_B=None)` (ABC from issue 00). Internally it: (i) EMA-smooths the demand, (ii) decides the phase flag, (iii) optionally fuses `d_B` (if provided, issue 06), then (iv) delegates to `_allocate(cell_ids, d, occupancy, B_total, phase)` (issue 03a). It never repeats any allocation mechanism.

**EMA smoothing.** Smooth the demand field over `τ_smooth` Controller steps:

```
d_smooth(t) = β · d_smooth(t − 1) + (1 − β) · d_raw(t)
where β = 1 − 1/τ_smooth, default τ_smooth = 3
```

**Spearman rank-stability gate.** Used to decide phase switch:

```
stable = SpearmanR(d_smooth(t), d_smooth(t − τ_smooth)) ≥ 0.9
         computed over shared cell_ids only (requires Partition's stable cell_ids)
```

Rank stability (not EMA magnitude stability) is the correct gate: pruning ranks by `s(a)`, and ranks can hold while magnitudes drift under global rescaling.

**Phase-flag decision:**

- **Phase 1 ("ramp"):** while `N_total < B_total` OR Spearman gate has not held for `k` consecutive steps.
  - Budget-aware crossing: at the step where growth would overshoot, scale grow deltas by `p = (B_total − N_total) / ΣΔ⁺` (proportional clamp, applied in 03a but gated by ramp phase).
- **Phase 2 ("steady"):** `N_total ≥ B_total` AND Spearman gate has held for `k` consecutive steps (default `k = 2–3`).
- **Plateau fallback:** if growth stalls below `B_total` (N_total unchanged for `k` steps while still in ramp, and Spearman holds), enter steady under the cap. Slack below `B_total` is honest, not padded.

**Cadence.** Demand field + Controller run every `N = update_interval = 100` iterations. Smoothing and Spearman operate in Controller-step time (not iteration time).

**State management.** The temporal layer owns persistent state across Controller steps: `d_smooth_prev`, `d_smooth_history` (buffer for Spearman comparison), `stable_count`, `current_phase`. State is reset on Controller activation.

## Acceptance criteria

- [ ] EMA: a step-change in raw demand reaches 63% of the new value within `τ_smooth` steps (EMA decay property verified)
- [ ] Spearman gate: identical demand fields → correlation = 1.0; reversed order → −1.0; random → near 0
- [ ] Spearman gate: computed over intersection of shared `cell_ids` only (Control Cells present in both steps)
- [ ] Phase switch: enters steady after `N_total ≥ B_total` AND Spearman gate holds for `k` consecutive steps
- [ ] Phase switch: does NOT enter steady if Spearman dips below 0.9 during the sustain window (counter reset)
- [ ] Phase switch: at full progressive unlock (new fine anchors appear), Spearman drops → gate resets → no premature Phase 2
- [ ] Plateau fallback: `N_total` unchanged for `k` steps while still ramp AND Spearman holds → enters steady (phase = "steady", cap at `N_total` < `B_total`)
- [ ] `plan()` public signature is `plan(cell_ids, d_A, occupancy, B_total, d_B=None)` — `phase` is determined internally, not passed by the caller; `d_B` is an optional second demand field (reserved for issue 06, ignored when None)
- [ ] Multi-step fixed-point: stable demand over consecutive steps → `δ ≈ 0` across all Control Cells
- [ ] Multi-step no-thrash: small random perturbation of demand → dead-band absorbs it → `δ` changes only for Control Cells above threshold
- [ ] All state is reset/re-initialised on Controller activation (not carried across training restarts)
- [ ] No CUDA import in `ocbgs/controller/` (local-testable invariant)
- [ ] Scenario-B guardrail: log whether Phase 2 was reached by `update_until` (spec §5 — a per-scene diagnostic; if a scene systematically fails to reach Phase 2 within the window, `update_until` must be raised for that scene and its baseline together, re-measuring `B_total`)

## Blocked by

- 03a-controller-static-allocator (extends BudgetController; allocator provides the internal `_allocate(cell_ids, d, occupancy, B_total, phase)` pure function that the temporal layer feeds into)
