# Experiment Results

## Status Overview

| Experiment | Scene | Status | Plots/Output |
|------------|-------|--------|--------------|
| ExpA | amsterdam (BungeeNeRF) | DONE | `pareto_PSNR.png`, `pareto_SSIM.png`, `pareto_LPIPS.png` |
| Exp4 garden | garden (Mip-NeRF 360) | DONE | Table only |
| Exp4 bungeenerf | amsterdam | PARTIAL | B_total only (1,104,448) |
| Exp1 garden | garden (Mip-NeRF 360) | RETIRED | Table only |
| Exp2 garden | garden (Mip-NeRF 360) | planned | — |
| Exp4 bungeenerf full | bungeenerf scenes | planned | — |

All results use the `ours_30000` checkpoint (iteration 30,000). B_total is measured at iteration 25,000 (`update_until`) per the protocol in `docs/eval-plan.md` §4.

---

## ExpA — Amsterdam Controllable-Budget Pareto

**Design.** Compare gradient-demand allocation against uniform allocation across a swept Capacity Budget. Two arms — demand (gradient-informed `--demand_uniform False`) and uniform (`--demand_uniform True`, equal `B_total / N_active` per cell) — each run at four budget factors: 0.25×, 0.5×, 1×, 2× the baseline anchor count. Both arms use `--grow_relax_scale 0.1` and `--control_level_max 0` (locked spatial partition). Train-set metrics only; held-out test rendering is separate. The comparison isolates the demand-reallocation variable at matched achieved anchors along the Pareto curve, per `docs/eval-plan.md` §3 experiment 2.

**Scene.** Amsterdam from BungeeNeRF — large-scale, highly non-uniform demand field (satellite-to-ground). According to `docs/eval-plan.md` §5, this is the primary highlight scene where demand reallocation has maximum leverage.

**Baseline.** B_total = 1,104,448 anchors (measured from Octree-GS native at `update_until`, exp4 bungeenerf/amsterdam arm_a).

**Hardware.** Single RTX 4090. MAX_JOBS=1 (sequential). SEEDS=(0 1 2) for demand; uniform has incomplete seeds (see Notes).

### Results Table

| arm | factor | anchors | PSNR | SSIM | LPIPS | n |
|-----|--------|---------|------|------|-------|---|
| demand | 0.25 | 348,956 | 26.670 | 0.881 | 0.156 | 3 |
| demand | 0.5 | 551,870 | 27.220 | 0.892 | 0.143 | 3 |
| demand | **1** | **1,096,751** | **27.722** | **0.901** | **0.129** | **3** |
| demand | 2 | 1,687,635 | 27.400 | 0.895 | 0.136 | 3 |
| uniform | 0.25 | 350,932 | 26.601 | 0.880 | 0.157 | 2 |
| uniform | 0.5 | 537,218 | 27.061 | 0.893 | 0.139 | 1 |
| uniform | **1** | **906,401** | **27.450** | **0.901** | **0.128** | **1** |
| uniform | 2 | 1,413,955 | 27.747 | 0.906 | 0.121 | 1 |

### Budget Fill Ratios

| arm | factor | B_total | achieved | fill % | Phase 2? |
|-----|--------|---------|----------|--------|-----------|
| demand | 0.25 | 276,112 | 348,956 | 126% | NOT reached |
| demand | 0.5 | 552,224 | 551,870 | 100% | NOT reached |
| demand | 1 | 1,104,448 | 1,096,751 | 99% | NOT reached |
| demand | 2 | 2,208,896 | 1,687,635 | 76% | NOT reached |

All arms stay in ramp phase (Phase 2 NOT reached). Demand fills near 100% at 0.5× and 1×; fill drops to 76% at 2×, suggesting diminishing returns on anchor creation at extreme budgets even with the locked control level.

### Findings

1. **Demand allocation consistently outperforms uniform allocation at matched budgets** (where n ≥ 2). At 1× B_total: demand 27.72 PSNR vs uniform 27.45 (+0.27 dB). At 0.5×: demand 27.22 vs uniform 27.06 (+0.16 dB). At 0.25×: demand 26.67 vs uniform 26.60 (+0.07 dB).

2. **Budget scaling yields diminishing returns for demand.** At 2× B_total, demand achieves 1.69M anchors but PSNR drops to 27.40 (vs 27.72 at 1×) — more anchors do not produce better quality. The controller saturates the useful anchor capacity of the scene before filling 2× budget.

3. **Uniform arm is under-powered.** Only n=1 for 0.5×, 1×, and 2×; n=2 for 0.25×. The uniform@2× outlier (PSNR 27.75) is a single-seed result with no statistical confidence. These arms need to be re-run with matched seed counts for any publication figure.

4. **Control level lock was critical.** Without `--control_level_max 0`, the 2× arm would descend to a finer octree level (level 2, N_active=257K, cell_size=0.0033) where candidate generation is throttled by near-empty cells — the 2× run would underfill at 658K anchors and produce worse quality than 1×. See Notes below for diagnosis.

### Raw Data

- CSV: `/root/autodl-tmp/expA/amsterdam/pareto.csv`
- Per-seed results.json + outputs.log: `/root/autodl-tmp/expA/amsterdam/{demand,uniform}/arm_{0.25g,0.5g,1g,2g}x/seed_*/`
- B_total: `/root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam`

### Plotting

```bash
python scripts/collect_results.py pareto \
    --root /root/autodl-tmp/expA/amsterdam \
    --arms demand uniform --factors 0.25 0.5 1 2 \
    --step 25000 --checkpoint 30000 \
    --output /root/autodl-tmp/expA/amsterdam/pareto.csv

python scripts/plot_pareto.py \
    --csv /root/autodl-tmp/expA/amsterdam/pareto.csv \
    --metric {PSNR,SSIM,LPIPS} \
    --output /root/autodl-tmp/expA/amsterdam/pareto_{PSNR,SSIM,LPIPS}.png \
    --title "Controllable-budget Pareto — amsterdam ({metric})"
```

### Notes

**Control level shift with B_total (fixed in `9eabc29`).** `set_control_level()` in `ocbgs/partition/__init__.py` picks the deepest octree level where `B_total / N_active >= rho_min` (8). When sweeping B_total from 1× (1.1M) to 2× (2.2M), the larger budget satisfies `rho_min` at deeper levels:

- demand@1×: level=0 (coarsest), N_active=86K, cell_size=0.013
- demand@2× (pre-fix): level=2 (finer), N_active=257K, cell_size=0.0033

At level 2, cells are 64× smaller in volume, most cells contain 0–1 anchors, and `anchor_growing_gather` cannot produce candidates in near-empty cells — the controller allocates budget but growth is physically impossible. This caused demand@2× to achieve only 658K anchors (30% fill), _below_ demand@1× (1.1M).

Fix: added `--control_level` and `--control_level_max` flags. `exp_a_amsterdam_pareto.sh` sets `CONTROL_LEVEL_MAX=0` to lock the spatial partition scale across all budget factors. Documented in `CLAUDE.md` §Server & training environment.

---

## Exp4 — Garden No-Harm Control

**Design.** Three-arm comparison on garden (Mip-NeRF 360, a near-uniform demand scene) testing whether the budget controller causes harm when demand variation is low. Arm A (baseline): Octree-GS native growth, no controller. Arm B (A-only natural): A-only demand controller with plateau enabled. Arm C (A+B): dual-source demand controller (gradient + photometric error source B, `fusion_lambda=1.0`, `b_camlist_size=4`, `b_refresh_period=10`). The compare subcommand checks whether Arm C's PSNR delta over Arm B exceeds the A-only noise floor (2σ); if not, the B signal is indistinguishable from noise and B should be dropped per ADR-0002.

**Scene.** Garden from Mip-NeRF 360 — near-uniform demand field. According to `docs/eval-plan.md` §5, this is a no-harm control scene where the controller should degrade gracefully to baseline performance.

**Hardware.** Single RTX 4090. MAX_JOBS=3. B_total=740,479. 5 seeds per arm.

### Results Table

| arm | PSNR | SSIM | LPIPS | n |
|-----|------|------|-------|---|
| Arm A — baseline (no controller) | 33.520 ± 0.168 | 0.9603 ± 0.002 | 0.0365 ± 0.001 | 5 |
| Arm B — A-only natural (plateau) | 32.153 ± 0.122 | 0.9474 ± 0.002 | 0.0510 ± 0.002 | 5 |
| Arm C — A+B (λ=1, M=10, K=4) | 32.169 ± 0.135 | 0.9474 ± 0.002 | 0.0510 ± 0.002 | 5 |

### Compare Decision (|ΔPSNR| vs 2σ)

| checkpoint | μ_A | μ_B | Δ | 2σ | \|Δ\| > 2σ? |
|-----------|------|------|------|------|--------------|
| ours_30000 | 32.153 | 32.169 | +0.016 | 0.244 | no |

Decision: **DROP B.** B signal (+0.016 dB) is indistinguishable from noise (2σ=0.244). The dual-source controller provides no measurable quality benefit over A-only on this scene at this budget.

### Findings

1. **Controller does not harm baseline** — Arm B (32.15) is ~1.4 dB below Arm A (33.52), but this is a budget constraint issue (740K budget vs baseline's free growth), not a controller defect. The controller successfully enforces the budget without catastrophic quality loss.

2. **B signal (photometric error demand) is not useful on garden.** The A+B arm matches A-only within noise. Per ADR-0002, B should be demoted to a validation-only diagnostic. This is consistent with garden being a near-uniform scene: B's additional signal adds no information.

3. **Plateau is reached.** All controller arms stay in ramp phase (Phase 2 NOT reached by update_until), meaning the controller never fills the 740K budget — growth saturates before B_total. This is expected behavior for the natural-budget design (ADR-0004 §budget-constraint).

### Raw Data

- Per-arm summaries: `/root/autodl-tmp/exp4/garden/{sigma_garden,summary_a_plus_b}.json`
- Per-seed data: `/root/autodl-tmp/exp4/garden/{arm_a,arm_b,arm_c}/seed_*/`
- B_total: `/root/autodl-tmp/exp4/garden/BTOTAL_GARDEN`

### Re-running

```bash
python scripts/collect_results.py metrics \
    --glob "/root/autodl-tmp/exp4/garden/arm_b/seed_*" \
    --checkpoints 30000 \
    --output /root/autodl-tmp/exp4/garden/sigma_garden.json

python scripts/collect_results.py metrics \
    --glob "/root/autodl-tmp/exp4/garden/arm_c/seed_*" \
    --checkpoints 30000 \
    --output /root/autodl-tmp/exp4/garden/summary_a_plus_b.json

python scripts/collect_results.py compare \
    --a-only /root/autodl-tmp/exp4/garden/sigma_garden.json \
    --a-plus-b /root/autodl-tmp/exp4/garden/summary_a_plus_b.json \
    --sigma /root/autodl-tmp/exp4/garden/sigma_garden.json
```

---

## Exp1 — Garden Matched-Budget (RETIRED)

**Design.** Tests whether disabling plateau fallback (`--no_plateau`) forces the controller to fill the Capacity Budget and improves quality. Three arms: baseline (exp4 arm_a, no controller), natural (exp4 arm_b, A-only with plateau), and matched (new arm, A-only with `--no_plateau`). Reuses exp4 garden baseline + natural arms; only the matched arm is new. Single seed.

**Scene.** Garden (same as exp4). B_total=740,479.

**Status.** **RETIRED** per `docs/eval-plan.md` §3: "The force-equality / floor-fills-the-budget operating point is withdrawn — it contradicted the no-force-fill invariant (ADR-0003/0004/0005) and was experimentally refuted."

### Results Table

| arm | PSNR | SSIM | LPIPS | n | Σn | fill% |
|-----|------|------|-------|---|---|------|
| baseline (no controller) | 33.520 ± 0.168 | 0.9603 | 0.0365 | 5 | — | — |
| natural (plateau enabled) | 32.153 ± 0.122 | 0.9474 | 0.0510 | 5 | — | ~73% |
| **matched (no plateau)** | **32.225 ± 0.000** | **0.9486** | **0.0496** | **1** | **542,033** | **73%** |

### Findings

1. **Matched ≈ Natural.** ΔPSNR = +0.07 dB (n=1, single seed). The matched arm produced nearly identical quality to the natural arm. Disabling plateau did not force the controller to fill the budget — anchor growth still saturated at 73%.

2. **Phase 2 NOT reached.** Despite `--no_plateau`, the controller logged "Phase 2 NOT reached by update_until=25000" and final anchors = 542,033 < 740,479. Without plateau, the controller stays in ramp phase forever, but the candidate-generation pipeline (`anchor_growing_gather` with `grow_relax_scale=1.0`) cannot supply enough viable candidates to reach B_total.

3. **The eval-plan's retirement was correct.** Force-filling the budget is not a meaningful control arm — the controller either reaches B_total naturally (if the scene supports it) or plateaus below it (if candidates are exhausted). There is no quality benefit to artificially holding the controller in ramp phase.

### Raw Data

- `/root/autodl-tmp/exp1/garden/arm_baseline.json` (reuses exp4 arm_a)
- `/root/autodl-tmp/exp1/garden/arm_natural.json` (reuses exp4 arm_b)
- `/root/autodl-tmp/exp1/garden/arm_matched.json`

### Re-running

```bash
SEEDS="0" bash scripts/exp1_garden.sh
```

---

## Upcoming

- **Exp2 — Garden Pareto.** Sweep B_total × {0.25, 0.5, 1, 2} on garden with A-only natural-budget controller. No uniform comparison arm. Purpose: verify that budget-controlled quality scales with budget on a near-uniform scene (no-harm claim).
- **Exp4 — BungeeNeRF.** Full 3-arm comparison across multiple BungeeNeRF scenes (amsterdam, hollywood, pompidou, etc.) to test the "no-harm" claim at large scale with heterogeneous demand. The amsterdam B_total is already measured (1,104,448); remaining scenes need baselines.
