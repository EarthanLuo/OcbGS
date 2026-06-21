# Exp 4 SOP — Source B KEEP/DROP Decision

## Purpose

Experiment 4 of `docs/eval-plan.md` §3. The question: does Source B (FastGS photometric residual) provide a quality gain that justifies its compute cost relative to A-only (gradient proxy)? The answer triggers the ADR-0002 fallback: KEEP B or demote to validation-only diagnostic.

## Background

- **ADR-0002 (§ Source B):** B is the photometric residual signal. Additive fusion `d = EMA[L1(d_A) + λ·L1(d_B)]`. Cost is forward renders on a subsampled `camlist` every M controller steps.
- **ADR-0002 (§ fallback):** If ablation shows B's quality gain does not justify its render cost, B is demoted to a validation-only diagnostic and the system ships A-only. The fallback is decided by **evaluation data**, not pre-committed in code.
- **Issue 06 smoke test:** B wired correctly — `s_B.shape = [N_anchors]`, `d_B_cells = 74194`, independent L1 norm verified (`d_A_norm_sum = d_B_norm_sum = 1.000`). But fused demand is extremely flat: `d_raw_range = [0.000, 0.002]` across ~74k cells. This is expected under L1 normalization and does **not** pre-judge usefulness — it means the signal is subtle, and statistical rigor is essential.
- **Scene allocation:**
  - **Garden** (m360) — near-uniform demand → no-harm control. Should match A-only; any degradation would be a regression.
  - **BungeeNeRF** (amsterdam / quebec / rome) — extreme satellite-to-ground scale variation → value judgment stage (ADR-0002 caveat: B exists to cover A's blind spots for fine/distant regions with small screen footprint).

## Noise floor

CUDA backward `atomicAdd` makes training non-bit-reproducible even with fixed seed (CLAUDE.md). Source B adds an independent `random.sample` for camlist selection. Every data point has intrinsic variance.

We measure **σ_PSNR / σ_SSIM / σ_LPIPS** across 5 seeds on the garden A-only (controller) baseline. σ defines the resolution floor: a |ΔPSNR| below 2σ cannot be distinguished from noise. **Per-checkpoint**, not terminal-only (B's effect may manifest early and A catch up later as atomicAdd variance accumulates). Checkpoints: 7000 / 15000 / 25000 / 30000.

## Statistical decision rule

| Outcome | Meaning |
|---------|---------|
| |ΔPSNR(A+B − A)| > 2σ_garden at any checkpoint | **KEEP B** — signal detectable above noise |
| |Δ| ≤ 2σ at all checkpoints | **DROP B** — indistinguishable from noise |
| Garden σ too large to resolve expected B effect | **INCONCLUSIVE(garden-fail)** — increase seeds or extend training |
| BungeeNeRF |Δ| ≤ 2σ after garden σ confirmed adequate | **DROP** — B failed in its intended non-uniform regime |

## Amortized cost reference

Production cadence: `update_interval = 100` (controller steps every 100 training iterations). Per-camera render ≈ 16.7ms (smoke: 66.7ms / 4 cams). **No frustum cull** (`visible_mask = None`) — cost is upper bound; adding `prefilter_voxel` per camera is the first optimization lever.

| arm | render/refresh | period (training iter) | amortized cost |
|-----|----------------|------------------------|----------------|
| M10-K4 | 66.7ms | 1000 | 0.43% |
| M10-K16 | 267ms | 1000 | 1.71% |
| M100-K4 | 66.7ms | 10000 | 0.043% |
| M100-K16 | 267ms | 10000 | 0.17% |

All arms lie below 2% amortized → the Pareto x-axis (training time) is compressed; the decision is purely about **quality significance**, not cost trade-off. B's cost is negligible regardless of configuration.

## Experiment plan

### Garden — no-harm control

Single command: `bash scripts/exp4_garden.sh`

Internal phases (fully automatic):

**Phase 1 — B_total measurement.** Arm A: Octree-GS native (`--no_controller`), 5 seeds. Extracts `total_points @ 25000` from TensorBoard events, writes `BTOTAL_GARDEN` file (5-seed mean). Skips if file exists.

**Phase 2 — Arms B and C in parallel.** Both use `--B_total $(cat BTOTAL_GARDEN)`. Arm B: A-only controller, 5 seeds (measures σ). Arm C: A+B controller (`--b_enabled --fusion_lambda 1.0 --b_camlist_size 4 --b_refresh_period 10`), 5 seeds (M10-K4: lowest cost, liveliest signal).

**Phase 3 — Auto-collect and compare.** Runs `scripts/collect_results.py metrics` on both arms, then `scripts/collect_results.py compare` to produce the decision table. Writes `sigma_garden.json` and `summary_a_plus_b.json`.

Output: `/root/autodl-tmp/exp4/garden/`

Verifies A+B does not degrade quality on a near-uniform scene (|ΔPSNR| ≤ 2σ at all checkpoints). If A+B exceeds A-only by more than 2σ, this is an unexpected finding worth investigating (B should not matter on near-uniform demand where A's gradient proxy is already accurate).

### BungeeNeRF — value judgment

Single command per scene: `bash scripts/exp4_bungeenerf.sh <scene>`

Default scenes: `amsterdam quebec rome`. Run one at a time (each uses all GPU memory).

Internal phases (fully automatic):

**Phase 1 — Quick B_total.** Arm A: single seed 0, Octree-GS native (`--no_controller`). Extracts `total_points @ 25000`, writes `BTOTAL_<SCENE>`. Skips if file exists.

**Phase 2 — Arms B and C in parallel.** Arm B: A-only controller, 5 seeds (measures σ). Arm C: A+B controller (`--b_enabled --fusion_lambda 1.0 --b_camlist_size 16 --b_refresh_period 10`), 5 seeds (M10-K16: strongest B signal).

**Phase 3 — Auto-collect and compare.** Same as garden. Writes `sigma_<SCENE>.json` and `summary_a_plus_b.json`.

Output: `/root/autodl-tmp/exp4/bungeenerf/<scene>/`

**Decision point:** if |ΔPSNR(A+B − A)| > 2σ_garden on any checkpoint → KEEP B. Otherwise → DROP B (B failed its core prediction of helping in non-uniform regions where A's gradient proxy is blind).

### Fidelity sweep (conditional)

Only if Step 4 KEEPs B. Sweep `b_camlist_size ∈ {4, 8, 16}` and `b_refresh_period ∈ {10, 100}` on the best BungeeNeRF scene. Answers: how sparse can B be while retaining the quality gain? Detailed in a follow-up issue — out of scope for this SOP.

## Metrics

| metric | source | purpose |
|--------|--------|---------|
| PSNR / SSIM / LPIPS | `results.json` (post-training eval pass) | primary quality |
| `total_points` | TensorBoard scalar per-checkpoint | B_total measurement + per-checkpoint Δ |
| `#anchors` (final) | `results.json` or TB `total_points @ 30000` | budget compliance |
| training wall-clock | shell `time` or TB `iter_time` | secondary |

## Tools

`scripts/collect_results.py` automates all data extraction and comparison. Three subcommands:

```bash
# Extract B_total from TensorBoard events
python scripts/collect_results.py total_points \
    --glob "/tmp/exp4/garden/arm_a/seed_*" --step 25000 \
    --aggregate mean --output-btotal BTOTAL_GARDEN

# Collect per-checkpoint PSNR/SSIM/LPIPS across seeds → summary JSON
python scripts/collect_results.py metrics \
    --glob "/tmp/exp4/garden/arm_b/seed_*" \
    --checkpoints 7000 15000 25000 30000 \
    --output sigma_garden.json

# Compare A+B vs A-only vs σ → print KEEP/DROP table
python scripts/collect_results.py compare \
    --a-only sigma_garden.json --a-plus-b summary_a_plus_b.json \
    --sigma sigma_garden.json
```

The shell scripts invoke the collector automatically in Phase 3. No manual TensorBoard reading required.

## Risks

- **Undersized `B_total` (Exp 2 Pareto sweep).** Scaling `B_total` to 0.25× baseline may cause `set_control_level` to fail silently (known bug: `.scratch/demand-driven-budget-reallocation/issues/bug-set-control-level-undersized-btotal.md`). Not triggered by Exp 4 (fixed `B_total` = baseline), but relevant for downstream Exp 2.
- **BungeeNeRF compute.** 5 seeds × 3 scenes × 2 arms = 30 full runs. Single RTX 4090 ≈ 5 days serial. Batch across GPUs if available; otherwise prioritize 2 scenes (amsterdam + quebec) first and run rome only if results are borderline.
- **B-arm additional variance.** `source_b.py:12` uses global `random.sample` for camlist → B-arm has an extra noise source beyond atomicAdd. σ for B-arm should be measured separately; A-only σ is the floor, not B-arm's σ.
- **Black background bias.** `evaluate_source_b` uses fixed black background. For white-background datasets, the error map will include spurious error from bg mismatch. Not applicable to garden or BungeeNeRF (both COLMAP, black bg compatible).

## Acceptance criteria

| # | Criterion | Evidence |
|---|-----------|----------|
| 1 | σ_PSNR/SSIM/LPIPS measured on garden A-only (controller) | `sigma_garden.json` from Phase 3 |
| 2 | Per-checkpoint σ at 7000 / 15000 / 25000 / 30000 | Metrics per checkpoint in `sigma_garden.json` |
| 3 | Garden A+B falls within ±2σ of A-only | |ΔPSNR| ≤ 2σ at all checkpoints (Phase 3 output) |
| 4 | BungeeNeRF B_total per scene measured | `BTOTAL_<SCENE>` files |
| 5 | BungeeNeRF |ΔPSNR| compared to 2σ_garden | >2σ → KEEP, ≤2σ → DROP (Phase 3 output per scene) |
| 6 | Decision documented with supporting data | This SOP updated with outcome |

## References

- ADR-0002 (Demand Producer — Error/Visibility signal)
- ADR-0004 (Budget Controller — conservation phases)
- `docs/eval-plan.md` §3 Experiments 3-4
- Issue 06 (`issues/06-source-b-fastgs-fusion.md`)
- Smoke log at iter 110: `s_B.shape=[496343]`, `d_B_cells=74194`, `d_raw_range=[0.000,0.002]`
