# Controllable Primitive Budget & Quality–Cost Pareto Front — Mainline Design

Date: 2026-06-26. Branch: `mainline-pareto-pivot`. Supersedes the *evaluation mainline* of the 2026-06-19 DDBR design spec (the architecture in that spec stands; only its headline claim and the matched-budget operating point are retired here).

## 1. Why this document exists (the pivot)

The 2026-06-19 DDBR spec framed the headline as **"same anchor budget, higher quality than Octree-GS"** — a *superiority* claim resting on a *matched-budget* operating point (`Σn ≡ B_total`, "floor fills the budget"). Two independent facts retired that framing:

1. **It contradicts the architecture's own no-force-fill invariant.** ADR-0003/0004/0005 all state that a candidate-limited cell simply grows fewer anchors and that `δ⁺` is an upper bound, not a mandatory quota — executed occupancy is `≤` planned `≤ B_total` *by design*. "Force equality / floor fills the budget" is the one place in the doc set that demanded the opposite.
2. **It was experimentally refuted.** The controller settles at ≈73% of `B_total` (structural, candidate-supply-limited, confirmed on a full 25k garden run), and at that operating point the A-only allocation is ≈ uniform. Matched-budget is unreachable without force-fill, and force-fill would pad the budget with low-value anchors — exactly what `CONTEXT.md` calls dishonest ("slack below `B_total` is honest, not padded with low-value anchors").

Crucially, the **graduation proposal never required a superiority claim.** Its quantified targets (proposal §五.2) are *maintain quality while cutting cost*: PSNR drop ≤ 0.5 dB, active Gaussians −50%, FPS ≥ 5×, training ≤ 1/3, flicker −50%. Its "main scientific problem five" is explicitly **controllable quality–cost trade-off (Pareto front)**. The old mainline was therefore solving a strictly harder, self-imposed problem than the degree requires.

This document re-bases the mainline on what the proposal actually asks for and what the architecture already supports: a **controllable primitive budget that produces a quality–cost Pareto front**, with demand-driven allocation and the photometric Source B as supporting evidence rather than as a baseline-beating wager.

## 2. Claims (tiered by load-bearing importance)

**Primary — capability / moat (not a superiority claim).** An LOD-structured, spatially demand-allocated *controllable* primitive budget that yields a quality–cost Pareto front. Octree-GS has no budget set-point (only growth/prune thresholds) and emits a single operating point; our controller takes a target `B_total` and traces a curve. The novelty is the *combination* — octree LOD structure **+** an explicit budget set-point **+** demand-driven spatial distribution of that budget across Control Cells — driving training reallocation through one unified budget. (3DGS-MCMC offers a global count cap but no LOD structure and no spatial demand allocation; Octree-GS offers LOD but no budget set-point; CDC-GS / Taming target other axes. The combination is the gap.)

**Secondary — demand beats uniform along the curve.** At matched *achieved* budget, demand-driven allocation (A = gradient, B = photometric, fused additively) lies on or above the uniform-allocation curve (uniform = Octree-GS's own spatial distribution). This is a *relative* claim against our own control, not against a tuned baseline; it only needs to hold, not to dominate.

**Tertiary — B is a targeted corrector of A's blind spot.** Source B surfaces under-fit fine/distant regions that the screen-space gradient proxy A systematically under-weights (2026-06-19 spec §4.1: A carries a screen-space scale bias against small-footprint regions). Shown on a near/far test-view split: A+B ≥ A-only on far/detail views, with near views unchanged.

**Efficiency reporting (rides on existing machinery).** FPS, active (opacity-masked) Gaussian count, training time — largely inherited from the Octree-GS LoD renderer; measured and reported against the maintain-quality bar, not engineered anew.

**Continuity (stretch).** CLoD-GS continuous opacity decay composed with reallocation, evaluated by flicker score / temporal LPIPS. This is the only tier that requires building rendering-side code; it is explicitly out of the graduation-critical path.

### Probe evidence already in hand (grounds the Secondary/Tertiary claims)

A per-Control-Cell instrumentation of `d_A` and `d_B` at the controller's real operating window (BungeeNeRF, iters 7000–11000, converged regime) established:

- **B is not flat.** Per-cell `d_B` is heavy-tailed: coefficient of variation ≈ 1.27, p99/p50 ≈ 9.8, ~98% of cells non-zero (amsterdam; all three scenes show CV > 1).
- **B is ~60% rank-independent of A.** Spearman(`d_A`, `d_B`) ≈ 0.58–0.69 across amsterdam/quebec/rome (R² ≈ 0.41 → ~59% of B's rank variance is unexplained by A); top-decile overlap ≈ 0.50 (half of A's highest-demand cells are *not* in B's top decile). Stable across the window — structure, not noise.

This refutes the earlier worry that B might be degenerate, and sizes the leverage as **moderate, not dramatic** — which is why the mainline does not stake the paper on B beating anything.

*Caveat carried into the spec:* the "B is more independent in larger scenes" reading (rome 185k cells, Spearman 0.577 lowest) is **not** yet a defensible trend — `n = 3`, and the naive `argsort` rank used in the probe is not tie-corrected, so more zero-`d_A` cells in large scenes can mechanically depress the correlation. Treat as suggestive; do not headline without a tie-corrected re-measure on more scenes.

## 3. Why the 73% undershoot is no longer a blocker

The Pareto curve is plotted on **achieved** budget, not the set-point. Sweeping `B_total` and recording where occupancy actually lands gives a valid curve regardless of undershoot — a 73%-of-set-point operating point is simply a point further left on the x-axis. The matched-budget claim was the *only* consumer of "hit `B_total` exactly"; with it retired, **no force-fill is required**, and the design returns to full consistency with ADR-0003/0004/0005. Fixing the undershoot becomes an *optional* lever (it would let us place curve points at chosen budgets rather than wherever they fall), not a prerequisite.

## 4. Experiments

Baselines (from eval-plan §1, unchanged): Vanilla 3DGS (capacity-agnostic reference), **Octree-GS = primary control = uniform allocation**, CLoD-GS (continuous-LOD comparison), FastGS (training-speed axis, optional). Octree-GS emits one point; ours sweeps.

**Exp A — Pareto front (the money figure).** Per scene × seeds, sweep `B_total ∈ {0.25, 0.5, 1, 2} ×` the per-scene Octree-GS baseline anchor count (measurement protocol = eval-plan §4, unchanged). For each run record *achieved* active anchors / opacity-masked Gaussians (x) vs PSNR/SSIM/LPIPS (y). Two curves: **demand (A+B)** and **uniform control**. Success: the demand curve lies on or above uniform across the swept range, and the curve spans budget points Octree-GS cannot address.

**Exp B — near/far view split (B's targeted value).** Partition test cameras by distance (or by per-view mean depth). Compare A+B vs A-only PSNR on the far/detail subset. Success: A+B ≥ A-only on far views, near views unchanged.

**Exp C — efficiency by-product.** Training time, FPS, active-Gaussian count, and the **A+B vs A-only training-time delta** (B's `2·|camlist|`-render cost must be shown, not hidden). This is the deciding datum for the ADR-0002 Source-B fallback: if B's quality gain does not justify its render cost, B is demoted to a validation-only diagnostic and the system ships A-only.

**Exp D — continuity (stretch).** Reallocation composed with CLoD-GS continuous opacity decay; flicker score / temporal LPIPS vs discrete LoD switching. Out of the graduation-critical path.

**Ablations (from eval-plan §3.3, retained).** Demand source: uniform (= Octree-GS) / gradient-only (`λ=0`) / gradient+photometric (`λ ∈ {0.5, 1.0}`) / photometric-only, holding the entire downstream pipeline identical (fairness condition). Controller knobs: `k_cap`, `θ_frac`, `r%`, `τ_smooth`, `k`. Reallocation headroom: `ρ_min`.

## 5. Success criteria (the bar, made explicit)

Aligned to the proposal's maintain-quality framing, not to beating a baseline:

- **Primary (capability):** a coherent Pareto curve exists and is controllable via `B_total`; at a chosen operating point quality is within **≤ 0.5 dB PSNR** of full Octree-GS while active Gaussians are materially lower. This is the load-bearing graduation result and depends only on the controller running and the sweep completing — not on any comparative win.
- **Secondary:** demand curve ≥ uniform curve (relative, moderate effect expected).
- **Tertiary:** A+B ≥ A-only on far views.
- A failure of Secondary or Tertiary does **not** sink the paper: the capability claim + efficiency + the honest B-leverage characterisation still constitute a defensible contribution (and trigger the ADR-0002 A-only fallback for B).

## 6. Relationship to existing docs (consistency map)

- **Retires** the *matched-budget* operating point: 2026-06-19 spec §6.3 line 309; eval-plan §3 line 28; and the "critical comparison = equal #anchors" framing (spec line 298, eval-plan line 14) → re-stated as "comparison along the Pareto curve at matched *achieved* #anchors, demand vs uniform."
- **Preserves unchanged** the no-force-fill invariant and all of ADR-0003/0004/0005 — now fully consistent, since the only clause that required force-fill is gone.
- **Adds** the near/far view-split evaluation (eval-plan §5 currently has only the capacity heatmap).
- **Keeps** `CONTEXT.md` as-is — its "honest slack, not padded with low-value anchors" already encodes the natural-budget stance.
- **Maps to the proposal:** Primary claim = proposal 创新点一 (unified voxel budget) + 主要问题五 (controllable trade-off / Pareto); Tertiary = proposal's Source-B / §4.1 screen-space blind-spot correction. The proposal needs no change — it already asks for maintain-quality + controllable budget.
- Old spec is retained (not deleted); a pointer to this document is added at its top.

## 7. Risks & fallbacks

- 🟡 **Demand may only tie uniform** (Secondary). Mitigated: capability + efficiency + far-view B carry the paper; tie is reported honestly.
- 🟡 **B may not justify its render cost** (Tertiary/Exp C). Mitigated: ADR-0002 fallback — demote B to validation-only diagnostic, ship A-only; the architecture already anticipates this.
- 🟡 **Rendering-side build cost** (Exp D continuity). Mitigated: it is stretch/out-of-critical-path; the graduation core is Exp A+B+C.
- 🟢 **rome "scaling with complexity" over-claim** — guarded in §2; not to be headlined without a tie-corrected re-measure.

## 8. Scope

- **In (graduation-critical):** Exp A (Pareto), Exp B (view split), Exp C (efficiency + B-cost), the demand-source and controller-knob ablations. Built largely on existing code (controller, Source B, `B_total` knob, `exp2_*_pareto.sh` and `collect_results.py` drafts).
- **Stretch (non-critical):** Exp D continuity (CLoD composition + flicker), optional undershoot fix to place curve points at chosen budgets.
- **Out:** the full 6-D voxel state vector, the five-state lifecycle machine, and large-scale (MatrixCity-class) scenes from the proposal — beyond the graduation core; revisit only if the core lands early.
