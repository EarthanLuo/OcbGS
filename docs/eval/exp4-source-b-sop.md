# Exp 4 SOP — Source B KEEP/DROP Decision

Experiment 4 of `docs/eval-plan.md` §3. The question: does Source B (FastGS photometric residual) provide a quality gain that justifies its compute cost relative to A-only (gradient-based)? Answer decides the ADR-0002 fallback: KEEP B or demote to validation-only diagnostic.

## Background

- **ADR-0002 (§ Source B, fallback):** B is the photometric residual signal. Additive fusion (`d = EMA[L1(d_A) + λ·L1(d_B)]`). Cost is forward renders on a subsampled `camlist` every M controller steps. If ablation shows benefit ≤ cost, B is demoted.
- **Smoke test (issue 06):** B wired correctly — `s_B.shape=[N_anchors]`, `d_B_cells=74194`, `d_A_norm_sum=d_B_norm_sum=1.000`. But fused demand extremely flat: `d_raw_range=[0.000, 0.002]` across ~74k cells (per-cell ~1.3e-5 average). This is expected under L1 normalization and does not pre-judge B's usefulness — it means the signal is subtle, and statistical rigor is essential.
- **Scene allocation:** garden (m360) = near-uniform demand → no-harm control (should match A-only). BungeeNeRF (amsterdam / quebec / rome) = extreme scale variation → value judgment stage (ADR-0002 caveat: B exists to cover A's blind spots for fine/distant regions with small screen footprint).

## Noise floor prerequisite

CUDA backward `atomicAdd` makes training non-bit-reproducible even with fixed seed (CLAUDE.md). Source B adds an independent `random.sample` for camlist selection. Every data point has intrinsic variance.

**Before any A vs A+B comparison, we must measure σ (standard deviation of quality metrics across seeds) on the garden A-only (controller) baseline.** σ defines the resolution floor: a |ΔPSNR| below 2σ cannot be distinguished from noise.

**Per-checkpoint, not terminal-only.** B's effect may manifest early and A catches up later (atomicAdd variance accumulates over training). Take σ at 7000 / 15000 / 25000 / 30000 iterations.

## Statistical decision rule

| Outcome | Meaning |
|---------|---------|
| |ΔPSNR(A+B − A)| > 2σ_garden at any checkpoint | KEEP B (signal detectable above noise) |
| |Δ| ≤ 2σ at all checkpoints | DROP B (cannot distinguish from noise) |
| Garden σ too large to resolve expected B effect | INCONCLUSIVE(garden-fail) → increase seeds / extend training |
| BungeeNeRF |Δ| ≤ 2σ | INCONCLUSIVE(bungee-no-delta) → DROP (B failed in its intended non-uniform regime) |

## Experiment plan

### Step 1 — Garden baseline (B_total + σ)

**Arm A: Octree-GS native** (`--no_controller`), 5 seeds. Measures `B_total = total_points @ update_until=25000`. Writes `BTOTAL_GARDEN` file.

**Arm B: A-only controller**, 5 seeds, `--B_total $(cat BTOTAL_GARDEN)`. Measures σ_PSNR / σ_SSIM / σ_LPIPS at per-checkpoint test iterations.

> Garden parameters: `--fork 2 --base_layer 10 --visible_threshold 0.0 --dist2level round --update_ratio 0.2`. `base_layer=10` matches upstream Mip-NeRF360 script (default -1 is adaptive); differing initialization would inflate σ.

Script: `scripts/exp4_garden.sh` — single command runs all three phases sequentially.

Dependency: Phase 1 (Arm A, `--no_controller`) completes first and auto-extracts `BTOTAL_GARDEN` via `scripts/collect_results.py total_points`. Phases 2 and 3 are then automatic.

### Step 2 — Garden no-harm control

A+B vs A-only: `--b_enabled --fusion_lambda 1.0 --b_camlist_size 4 --b_refresh_period 10` (M10-K4: lowest cost, liveliest signal). 5 seeds each. `--B_total $(cat BTOTAL_GARDEN)`. Verifies A+B does not degrade quality relative to A-only on a near-uniform scene (±2σ).

Script: `scripts/exp4_garden.sh` (Phase 2: Arm C vs Arm B auto-compare)

### Step 3 — BungeeNeRF baseline (B_total + σ)

A-only controller, 5 seeds per scene (amsterdam / quebec / rome). Full BungeeNeRF parameters: `--progressive True --fork 2 --base_layer 10 --levels -1 --dist_ratio 0.99 --init_level -1 --extra_ratio 0.25 --extra_up 0.01`. Measures σ and `B_total` per scene.

Script: `scripts/exp4_bungeenerf.sh` (Phase 1 + 2). Single command per scene — Phase 1 auto-extracts `BTOTAL_<SCENE>` from a seed-0 run.

### Step 4 — BungeeNeRF value judgment

A+B (λ=1, M=10, K=16): strongest B signal. 5 seeds per scene vs A-only baseline. Same `scripts/exp4_bungeenerf.sh` (Phase 2 Arm C + Phase 3 auto-compare).

**Decision point:** if |ΔPSNR(A+B − A)| > 2σ_garden on any checkpoint → KEEP B. Otherwise → DROP B (B failed its core prediction of helping in non-uniform regions).

### Step 5 — Fidelity sweep (conditional, only if Step 4 KEEPs B)

Sweep `b_camlist_size ∈ {4, 8, 16}` and `b_refresh_period ∈ {10, 100}` on the best BungeeNeRF scene. Answers: how sparse can B be while retaining the quality gain? Provides the cost-quality frontier for production deployment. Out of scope for this SOP — detailed in a follow-up issue.

## Amortized cost reference

Production cadence: `update_interval=100`. Per-camera render ≈ 16.7ms (smoke: 66.7ms / 4 cams). No frustum cull (`visible_mask=None`) → cost is upper bound; adding `prefilter_voxel` per camera is the first optimization lever.

| arm | render/refresh | period (iter) | amortized |
|-----|----------------|---------------|-----------|
| M10-K4 | 66.7ms | 1000 | 0.43% |
| M10-K16 | 267ms | 1000 | 1.71% |
| M100-K4 | 66.7ms | 10000 | 0.043% |
| M100-K16 | 267ms | 10000 | 0.17% |

All arms lie <2% amortized → the Pareto x-axis (training time) is compressed; the decision is purely about quality significance, not cost trade-off. B's cost is negligible regardless of configuration.

## Metrics

Primary: PSNR / SSIM / LPIPS from `results.json` (post-training eval pass on test set). Secondary: `#anchors` (final), training wall-clock time. `total_points` from TensorBoard scalars at each test iteration for per-checkpoint Δ.

## Risks

- **Undersized `B_total` (Exp 2 Pareto sweep):** scaling `B_total` to 0.25× baseline may cause `set_control_level` to fail silently (known bug: `bug-set-control-level-undersized-btotal.md`). Not triggered by Exp 4 (fixed `B_total` = baseline), but relevant for downstream Exp 2.
- **BungeeNeRF scale:** 5 seeds × 3 scenes × 2 arms = 30 full runs. Single RTX 4090 serial ≈ 5 days. Batch across GPUs if available; otherwise prioritize 2 scenes (amsterdam + quebec) first and run rome only if results are borderline.
- **B-arm variance:** `source_b.py:12` uses global `random.sample` for camlist → B-arm has an additional noise source beyond A-only's atomicAdd. σ for B-arm should be measured separately; A-only σ is the floor, not B-arm's σ.
- **Black background bias:** `evaluate_source_b` uses fixed black background. For white-background datasets (NeRF synthetic), error map will include spurious error from bg mismatch. Not applicable to garden/BungeeNeRF (both COLMAP, black bg compatible).

## Tools

`scripts/collect_results.py` automates all data extraction and comparison:

```
# Read TB total_points at given iterations
python scripts/collect_results.py total_points \
    --glob "<output>/arm_a/seed_*" --step 25000 \
    --aggregate mean --output-btotal BTOTAL_GARDEN

# Collect PSNR/SSIM/LPIPS from results.json across seeds
python scripts/collect_results.py metrics \
    --glob "<output>/arm_b/seed_*" \
    --checkpoints 7000 15000 25000 30000 \
    --output sigma_garden.json

# Compare A+B vs A-only vs sigma → KEEP/DROP
python scripts/collect_results.py compare \
    --a-only sigma_garden.json --a-plus-b summary_a_plus_b.json \
    --sigma sigma_garden.json
```

All shell scripts invoke the collector automatically in Phase 3. No manual TB reading required.

## Acceptance criteria

| # | Criterion | Evidence |
|---|-----------|----------|
| 1 | σ_PSNR/SSIM/LPIPS measured on garden A-only (controller) | `results.json` across 5 seeds |
| 2 | Per-checkpoint σ at 7000 / 15000 / 25000 / 30000 | TB `total_points` at each checkpoint |
| 3 | Garden A+B falls within ±2σ of A-only | |ΔPSNR| ≤ 2σ at all checkpoints |
| 4 | BungeeNeRF B_total per scene measured | `total_points @ 25000` |
| 5 | BungeeNeRF |ΔPSNR| compared to 2σ_garden | >2σ → KEEP, ≤2σ → DROP |
| 6 | Decision documented with supporting data | Update this SOP with outcome |

## References

- ADR-0002 (Demand Producer — Error/Visibility signal)
- ADR-0004 (Budget Controller — conservation phases)
- `docs/eval-plan.md` §3 Experiments 3-4
- Issue 06 (`issues/06-source-b-fastgs-fusion.md`)
- Smoke log at iter 110: `s_B.shape=[496343]`, `d_B_cells=74194`, `d_raw_range=[0.000,0.002]`
