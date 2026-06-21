# Issue: Source B (FastGS photometric residual) + A+B additive fusion

**Status:** DONE

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

- [x] `ErrorVisibilityDemand` can produce both `s_A(a)` (gradient-based) and `s_B(a)` (photometric) as separate tensors — `s_A` from `training_statis` accumulators, `s_B` from forward renders over a subsampled camlist. **Two deviations from literal text, both per documented design decisions:** (1) `s_B` is produced by a *separate* `PhotometricDemand` producer, not by extending `ErrorVisibilityDemand` — ADR-0002 (§ "Signal architecture") explicitly permits "a separate producer instance"; (2) cost is **`1·|camlist|`** forward renders, not `2·|camlist|` — the per-anchor attribution is a CUDA-free L1 error-map scatter (ADR-0001: rasterizer never modified), so it does not replicate FastGS's second `metric_map`/atomicAdd pass. Confirmed by `render_ms ≈ 16 ms/camera × 4 = 66.7 ms` (a 2-pass design would be ~132 ms).
- [x] `s_B(a)` shape matches `[N_anchors]`, all values ≥ 0 — server smoke: `s_B.shape=torch.Size([496356])`, `s_B.min()=0.0000`
- [x] `d_B(v)` is computed by passing `s_B(a)` through the same `Partition.reduce()` as Source A — no code duplication in Partition — server smoke: `d_B_cells=74194` from the reused `Partition.reduce`
- [x] A+B fusion in Controller: `d_A` and `d_B` are L1-normalised independently, then added with λ weighting — local scale-invariance unit test + server fusion log `d_A_norm_sum=1.000 d_B_norm_sum=1.000`
- [x] `λ = 0` → fused `d` equals `normalize(d_A)` (A-only recovery verified by unit test) — local unit test (λ=0 → output equals A-only `_allocate`)
- [x] `λ = 1` → fused `d` equally weights both signals — local unit test + server log `lambda=1.00`
- [x] Additive fusion: a Control Cell with `d_A = 0` but `d_B > 0` gets non-zero fused demand (B can independently light up a Control Cell A missed) — local contrast unit test: `c_target[d_A=0 cell]` is strictly larger with `d_B>0` than with `d_B=0` (kills the multiplicative implementation)
- [x] Controller's public `plan()` signature is unchanged from issue 03b — fusion is internal — `plan(cell_ids, d_A, occupancy, B_total, d_B=None)` unchanged; the d_A/d_B alignment + L1-norm + additive fusion live inside `plan` and the new `align_demand_b` helper
- [x] B render cost is explicitly logged (GPU time in ms per B evaluation, as % of total training time per Controller step) — server smoke: `render_ms=66.7 amortized_pct=4.28%` (M=10; ≈0.4% at the default M=100)
- [x] Controller `M` (B refresh period): `d_B` is held constant between B evaluations; `d_A` updates every Controller step — server smoke: `[SOURCE_B] refresh` fires once (iter 110, `_b_step=10`), then `[FUSION]` runs every Controller step (120–180) with the held cached `d_B`; local E1/E3 unit tests lock the hold/refresh cache semantics
- [x] `|camlist|` and `M` knobs are exposed in the training config — `b_camlist_size`, `b_refresh_period`, plus `fusion_lambda` and `b_enabled` in `arguments/__init__.py` (visible in the run's args Namespace)

## Verification

**Local (pure-logic, CUDA-free) — 22 unit tests, all green.** Run with `pytest` (no GPU required):
- `accumulate_view` (10): projection convention (no-transpose, pinned by a non-identity proj matrix), grid_sample y-axis alignment, behind-camera / on-plane `w≤0` guard (no NaN), masked→full→global two-level index scatter, opacity shape-broadcast guard, empty mask, zero radii, non-negativity, linearity.
- `evaluate_source_b` orchestration (3): M-gate hold vs refresh, `zero_`/re-alloc of the accumulator, refresh→hold cross-step cache identity.
- Fusion layer (9): `align_demand_b` (new-cell→0, stale-cell drop), λ=0 A-only recovery, λ=1 equal weight, **independent L1-norm scale-invariance** (d_B scaled 100× leaves `plan` output unchanged), **additive d_A=0 lighting** (contrast `with_b > without_b`), `d_B=None` sentinel, zero-sum safety.

**Server (RTX 4090, sm_89) — end-to-end smoke, no crash.**
```
python train.py -s /root/autodl-tmp/m360/garden --ds 8 -m .../verify_run2 \
  --iterations 200 --start_stat 5 --update_from 10 --update_interval 10 --update_until 180 \
  --B_total 500000 --b_enabled --fusion_lambda 1.0 --b_camlist_size 4 --b_refresh_period 10
```
Evidence:
- `[SOURCE_B] refresh | s_B.shape=torch.Size([496356]) s_B.min()=0.0000 d_B_cells=74194 render_ms=66.7` — AC#2/#3 (real render, shape `[N_anchors]`, ≥0, reduced to cell-level d_B).
- `[FUSION] d_B non-null | d_A_norm_sum=1.000 d_B_norm_sum=1.000 lambda=1.00 d_raw_range=[0.000,0.002]` — AC#4/#6 (independent L1-norm + additive fusion live, λ wired through to the controller).
- `[SOURCE_B] iteration=110 render_ms=66.7 step_fwdbwd_ms=15.6 amortized_pct=4.28%` — AC#9 (cost honestly amortised: ~4% at M=10, ~0.4% at the default M=100; hold steps log nothing).
- `refresh` fires once (iter 110), `[FUSION]` runs every Controller step 120–180 — AC#10 (d_B held; d_A updates every step).
- `[CONTROLLER] final anchors=497887 <= B_total=500000: True` — budget conserved with B active.

**Known limitations (not blocking; for the eval phase):**
- This smoke proves *wiring*, not *value*. There is no A-only (λ=0) control run and only 200 iterations, so it does not show B improves quality. The fused signal is very diffuse (`d_raw_range=[0,0.002]` over 74194 cells). Whether B meaningfully redistributes capacity is the A-vs-A+B comparison in `docs/eval-plan.md` (Exp 4) and is the ADR-0002 data-driven KEEP/DROP decision. **DONE = implementation complete, not B retained.**
- B renders with `visible_mask=None` (all anchors, no `prefilter_voxel` frustum cull) — a deliberate choice to save one cull render. Negligible at the default M, but the first cost lever if eval pushes `|camlist|` up or M down.

## Blocked by

- 05-adjust-anchor-integration (needs full pipeline running end-to-end to integrate B into, and to measure B's render cost vs. quality gain)
