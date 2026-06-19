# Evaluation Plan — Demand-Driven Budget Reallocation

This document expands §6 of the DDBR design spec (`docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`) and closes the `B_total` measurement and Pareto-sweep forward-references from ADR-0004 and ADR-0005. It is an experiment plan, not an architecture decision record.

## 1. Baselines

| Method | Role |
|--------|------|
| Vanilla 3DGS (gsplat) | capacity-agnostic reference |
| **Octree-GS** | **primary base / control** (= uniform allocation; ours = + demand reallocation) |
| CLoD-GS | continuous-LOD comparison |
| FastGS | training-speed axis (optional) |

The critical comparison is vs Octree-GS at equal #anchors, isolating the demand-reallocation variable.

## 2. Metrics

| category | metric | role |
|---|---|---|
| Quality | PSNR / SSIM / LPIPS | primary |
| Budget | **#anchors** | controlled variable |
| Derived | rendered #Gaussians (opacity-masked), memory, render FPS, training time | reported, not controlled |

## 3. Experiments

**1 — Main comparison: two operating points, both on the Pareto curve.**

- **Matched-budget:** force equality `Σn ≡ B_total = Octree-GS final #anchors` (plateau off; floor fills the budget). Strictly equal #anchors → "same budget, higher quality."
- **Natural-budget:** the cap `Σn ≤ B_total` (plateau allowed). #anchors `≤` baseline at higher quality → "less budget, higher quality" (stronger claim; #anchors reported explicitly).

**2 — Money figure: Pareto curve.** Sweep `B_total ∈ {0.25, 0.5, 1, 2}× baseline` → quality. Claim: our curve dominates across budgets.

**3 — Ablations.**

- **Demand source:** uniform (= Octree-GS) / gradient-only (`λ=0`) / gradient + photometric (`λ ∈ {0.5, 1.0}`) / photometric-only. Plus B-cost knobs `|camlist|` and `M` (cost–quality trade-off — see Exp 4).
  - **Fairness condition:** hold the entire downstream pipeline identical across arms (`τ_smooth`, cadence, controller L1 normalisation, floor/cap) and vary only the raw signal source — so any quality delta is attributable to the signal's informativeness, not to an incidental change of distribution shape. No per-producer distribution alignment (that would erase the shape this ablation measures).
- **Conservation:** hard / soft / none.
- **Reallocation headroom:** sweep `ρ_min` (mean occupancy; `control_level` is derived from `ρ_min` + `B_total`, ADR-0003) — granularity vs per-Control-Cell-headroom trade-off.
- **Controller knobs:** `k_cap` (cap multiple), `θ_frac` (dead-band), `r%` (rate-limit), `τ_smooth` (shared smoothing/gate horizon), `k` (gate sustain count). Defaults in ADR-0004.
- **Optional (render-only):** reallocation + CLoD-GS continuous opacity decay — composability check; expected to barely move still-image metrics.

**4 — By-product: training time / FPS (compute-saving corollary).** Includes the **A+B vs A-only training-time delta** — B's `2·|camlist|`-render cost must be shown, not hidden. This is the deciding data point for the Source B fallback (ADR-0002): if the quality gain does not justify the render cost, B is demoted to a validation-only diagnostic and the system ships A-only.

**5 — Qualitative: per-Cell Target Capacity heatmap.** Shows capacity flowed to high-detail regions. Also previews the future semantic version (swap heatmap for semantic ROI).

## 4. `B_total` measurement procedure

`B_total` is defined as the Capacity Budget the Controller conserves, conceptually equal to the baseline's final anchor count at `update_until` — the architectural definition and its coupling to the Controller window are in ADR-0004. This section specifies the measurement protocol.

**Protocol.**

- Run the **Octree-GS baseline** once per scene with a **fixed random seed**.
- `B_total` = the anchor count at **`update_until`** (iteration 25000 under default Octree-GS schedule). At this point `adjust_anchor` stops, so the count is **frozen** — it is the final anchor count for the baseline.
- **Not the PSNR-best checkpoint.** Selecting `B_total` from the best-PSNR checkpoint would couple the budget to the baseline's metric, breaking the pure capacity-pairing isolation between our method and the baseline. The `update_until` freeze point is metric-independent and unambiguous.
- Report the exact per-scene `B_total` value.
- For the Pareto sweep (Experiment 2): scale from this per-scene baseline value.

## 5. Datasets

**Selection principle:** the highlight scene is chosen to maximise demand-field non-uniformity — the regime where reallocation has the most leverage. A uniform demand field gives `c*(v) → B_total / N_active` (the Controller degrades to uniform allocation), so leverage grows with skew.

- **Standard benchmarks:** Mip-NeRF360, Tanks & Temples, Deep Blending — the expected benchmark tables (main comparison + Pareto + ablations). These sit at the near-uniform end, doubling as the **graceful-degradation / no-harm control**: we should match the baseline where there is nothing to reallocate.
- **Large-scale highlight: BungeeNeRF** (multi-scale, satellite→ground). Its extreme scale variation makes the demand field highly non-uniform, the strongest stage for the budget-reallocation story. Supported out of the box by the Octree-GS base (`train_bungeenerf.sh`).
- **MatrixCity dropped.** Aerial city capture is single-scale and geometrically regular ⇒ near-uniform demand ⇒ the method has no leverage. The standard benchmarks already cover the uniform end, so a second large uniform scene adds cost without evidence.

The narrative closes in one line: **the more skewed the demand, the more we win** — near-uniform standard scenes ⇒ ≈ baseline (no harm); non-uniform BungeeNeRF ⇒ ≫ baseline.

## 6. Reproducibility

- **Environment:** `environment.yml` (loose pins, tolerant of the server's arbitrary PyTorch version) + a one-shot `setup.sh` (create env, build Octree-GS CUDA submodule).
- **Seeds:** fixed random seeds — including the seed for the baseline run that defines each scene's `B_total` (Section 4).
- **Config recording:** Octree-GS `arguments/` config system records every experiment's settings.
- **GPU usage:** training is single-GPU per job by design. "Arbitrary GPU count" means job-level parallelism — one (scene × budget) combination per GPU, fanning the Pareto sweep and multi-scene runs across GPUs.
