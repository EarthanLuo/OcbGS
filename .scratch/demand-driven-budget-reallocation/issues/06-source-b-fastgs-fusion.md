# Issue: Source B (FastGS photometric residual) + A+B additive fusion

**Status:** ready-for-agent

## What to build

Add Source B — the FastGS photometric residual — as a second, independent per-anchor demand signal, and wire it through the full pipeline for A+B additive fusion in the BudgetController. Source B conforms to the same `DemandProducer` contract as Source A: per-anchor, raw, partition-agnostic.

**Source B production — `ErrorVisibilityDemand` extension:**

Port `compute_gaussian_score_fastgs` from the FastGS reference (`refered_repo/FastGS/utils/fast_utils.py:45-105`):

```
B(a) = photometric_loss(a) × accum_loss_counts(a)
```

- `photometric_loss(a)`: per-Gaussian contribution to the photometric error (L1 + SSIM), computed from `compute_photometric_loss` per camera in a subsampled `camlist`
- `accum_loss_counts(a)`: per-Gaussian count of views where the anchor contributed to high-error pixels, accumulated across the `camlist`
- **Cost:** ~`2 · |camlist|` forward renders per B evaluation (forward-only, no backprop). Two knobs bound cost: `|camlist|` (camera subsample) and `M` (Controller steps between B refreshes; `d_B` is held between evaluations)

**Adaptation to octree-anchor model.** The FastGS reference operates on standard 3DGS Gaussians. Adapt the computation to Octree-GS anchors: render the anchor set, accumulate per-anchor photometric error and visibility counts, produce a `[N_anchors]` tensor.

**A+B independent reduction.** `s_B(a)` is reduced independently by the same `Partition.reduce()` (issue 02) to produce `d_B(v)`. No change to Partition.

**A+B additive fusion in BudgetController (ADR-0004 § fusion):**

```
d(v) = EMA_τsmooth[ normalize(d_A) + λ · normalize(d_B) ]
```

- **Additive, not multiplicative.** Multiplicative `d_A · (1 + α · d_B)` gates B by A: where the gradient proxy is blind (`d_A ≈ 0`), B cannot raise demand — defeating the correction B exists to provide (ADR-0002).
- **L1-normalised per signal** — so `λ` is a meaningful relative weight and neither signal swamps the other.
- **`λ = 0` recovers A-only** (the system ships with `λ = 0` behaviour when B is disabled or ablated).

**Fusion implementation.** The Controller's public `plan(cell_ids, d_A, occupancy, B_total, d_B=None)` already reserves `d_B` as an optional parameter (issue 00). When `d_B` is None, the Controller uses A-only (λ=0, no-op); when provided, the temporal layer (03b) fuses A+B before delegating to `_allocate`. The signature does not change — only the caller (05) begins passing `d_B`.

**Ablation axes (exposed as config knobs):**
- `λ`: fusion weight (0 = A-only)
- `|camlist|`: camera subsample size for B
- `M`: B refresh period in Controller steps

**Data-driven fallback (ADR-0002 § fallback).** If ablation shows B's quality gain does not justify its render cost, B is demoted to a validation-only diagnostic and the system ships A-only. The fallback is decided by evaluation data, not pre-committed in code.

## Acceptance criteria

- [ ] `ErrorVisibilityDemand` can produce both `s_A(a)` (gradient-based) and `s_B(a)` (photometric) as separate tensors — `s_A` from `training_statis` accumulators, `s_B` from `2·|camlist|` forward renders over a subsampled camlist
- [ ] `s_B(a)` shape matches `[N_anchors]`, all values ≥ 0
- [ ] `d_B(v)` is computed by passing `s_B(a)` through the same `Partition.reduce()` as Source A — no code duplication in Partition
- [ ] A+B fusion in Controller: `d_A` and `d_B` are L1-normalised independently, then added with λ weighting
- [ ] `λ = 0` → fused `d` equals `normalize(d_A)` (A-only recovery verified by unit test)
- [ ] `λ = 1` → fused `d` equally weights both signals
- [ ] Additive fusion: a cell with `d_A = 0` but `d_B > 0` gets non-zero fused demand (B can independently light up a cell A missed)
- [ ] Controller's public `plan()` signature is unchanged from issue 03b — fusion is internal
- [ ] B render cost is explicitly logged (GPU time in ms per B evaluation, as % of total training time per Controller step)
- [ ] Controller `M` (B refresh period): `d_B` is held constant between B evaluations; `d_A` updates every Controller step
- [ ] `|camlist|` and `M` knobs are exposed in the training config

## Blocked by

- 05-adjust-anchor-integration (needs full pipeline running end-to-end to integrate B into, and to measure B's render cost vs. quality gain)
