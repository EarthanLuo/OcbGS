# Controllable Primitive Budget & Quality–Cost Pareto Front — Mainline Design

Date: 2026-06-26. Branch: `mainline-pareto-pivot`. Supersedes the *evaluation mainline* of the 2026-06-19 DDBR design spec (the architecture in that spec stands; only its headline claim and the matched-budget operating point are retired here).

## 1. Why this document exists (the pivot)

The 2026-06-19 DDBR spec framed the headline as **"same anchor budget, higher quality than Octree-GS"** — a *superiority* claim resting on a *matched-budget* operating point (`Σn ≡ B_total`, "floor fills the budget"). Two independent facts retired that framing:

1. **It contradicts the architecture's own no-force-fill invariant.** ADR-0003/0004/0005 all state that a candidate-limited cell simply grows fewer anchors and that `δ⁺` is an upper bound, not a mandatory quota — executed occupancy is `≤` planned `≤ B_total` *by design*. "Force equality / floor fills the budget" is the one place in the doc set that demanded the opposite.
2. **It was experimentally refuted.** The controller settles at ≈73% of `B_total` (structural, candidate-supply-limited, confirmed on a full 25k garden run), and at that operating point the A-only allocation is ≈ uniform. Matched-budget is unreachable without force-fill, and force-fill would pad the budget with low-value anchors — exactly what `CONTEXT.md` calls dishonest ("slack below `B_total` is honest, not padded with low-value anchors").

Crucially, the **graduation proposal never required a superiority claim.** Its quantified targets (proposal §五.2) are *maintain quality while cutting cost*: PSNR drop ≤ 0.5 dB, active Gaussians −50%, FPS ≥ 5×, training ≤ 1/3, flicker −50%. Its "main scientific problem five" is explicitly **controllable quality–cost trade-off (Pareto front)**. The old mainline was therefore solving a strictly harder, self-imposed problem than the degree requires.

This document re-bases the mainline on what the proposal actually asks for and what the architecture already supports: a **controllable primitive budget that produces a quality–cost Pareto front**, with demand-driven allocation and the photometric Source B as supporting evidence rather than as a baseline-beating wager.

### Two codebases — and the gap the graduation core must close

A prior implementation, **OCB-3DGS-HR** (gsplat-based, `D:\01_Projects\Active\Paper-research-1\OCB-3DGS-HR`), already built the *rendering-side* of the proposal — the full 6-D voxel state, the five-state lifecycle machine, the two-layer LoD renderer, and a budget sweep — and produced a clean **render-time** quality–cost curve (`my_output/M3_results.md`, lego: 72.9% Gaussians → PSNR 30.12, FPS 31→187). That result is real but **moat-thin on its own**: applying a budget at render time to an already-trained model is post-hoc LoD/pruning (close to LightGaussian / Octree-GS LoD) — it can only *degrade gracefully*, never make a region better than the trained model already is.

The current **OcbGS** codebase (vendored Octree-GS) was started precisely to build the *training-side* — demand-driven budget allocation *during* training — which is where the novelty/moat lives, because "better where it matters at equal budget" is fundamentally a training-time act (it changes where capacity went), only ever *observed* by rendering. The two codebases are complementary halves: render-side done (OCB-3DGS-HR, now a **comparison baseline**), training-side the live bet (OcbGS, this mainline). The proposal's soul — **one budget governing both training and rendering (训练-渲染一体化)** — is the unbuilt seam between them; closing it (at the level scoped in §2/§8) is a graduation-core deliverable, not a stretch goal.

## 2. Claims (tiered by load-bearing importance)

**Primary — controllable, training-side budget with a unified train/render decision (capability / moat, not a superiority claim).** A single Capacity Budget governs the *training-time* anchor population — demand-driven reallocation across Control Cells under `B_total` — and that same LOD-structured population is what the renderer activates per view. The budget therefore drives **both ends through the shared octree anchor structure**: training shapes which anchors exist (and at which level), rendering activates a distance-subset of exactly that population. This closes the proposal's "information gap" (训练-渲染一体化) *structurally* — one demand-shaped, budget-bounded anchor population serves both. Sweeping `B_total` traces a **training-side** quality–cost Pareto front (the model is *built for* each budget, not pruned after). The novelty is the combination — octree LOD structure **+** an explicit budget set-point **+** demand-driven spatial distribution during training — which no single prior method offers: 3DGS-MCMC has a global count cap but no LOD structure and no spatial demand allocation; Octree-GS has LOD but no budget set-point; CDC-GS / Taming target other axes; and post-hoc render-time budgeting (OCB-3DGS-HR M3) only degrades a fixed model.

**Secondary — demand dominates uniform, especially at low budgets (graceful-degradation dominance).** At matched *achieved* budget, demand-driven allocation (A = gradient, B = photometric, fused additively) lies on or above the uniform-allocation curve (uniform = Octree-GS's own spatial distribution). The expected and most defensible form of this is **at the low-budget end**: as the budget shrinks, demand keeps capacity in the regions that need it (far/detail) while uniform lets them collapse — so our curve stays up where uniform falls off. This is the operational meaning of "better where it matters": not beating a full baseline, but *degrading more gracefully* because the surviving budget is spent where error is. It is a *relative* claim against our own uniform control; it only needs to hold, not to dominate everywhere.

**Tertiary — B is a targeted corrector of A's blind spot.** Source B surfaces under-fit fine/distant regions that the screen-space gradient proxy A systematically under-weights (2026-06-19 spec §4.1: A carries a screen-space scale bias against small-footprint regions). Shown on a near/far test-view split: A+B ≥ A-only on far/detail views, with near views unchanged.

**Efficiency reporting (rides on existing machinery).** FPS, active (opacity-masked) Gaussian count, training time — largely inherited from the Octree-GS LoD renderer; measured and reported against the maintain-quality bar, not engineered anew.

**Continuity (stretch).** CLoD-GS continuous opacity decay composed with reallocation, evaluated by flicker score / temporal LPIPS. This is the only tier that requires *new* rendering-side code beyond Octree-GS's native LoD renderer; it is explicitly out of the graduation-critical path.

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

**Exp A — Training-side Pareto front (the money figure).** Per scene × seeds, sweep `B_total ∈ {0.25, 0.5, 1, 2} ×` the per-scene Octree-GS baseline anchor count (measurement protocol = eval-plan §4, unchanged). **Each point is a full budget-guided training run** — the model is *built for* that budget, not pruned afterwards. Record *achieved* active anchors / opacity-masked Gaussians (x) vs PSNR/SSIM/LPIPS (y). Two curves: **demand (A+B)** and **uniform control**. Success: the demand curve lies on or above uniform across the swept range — most importantly at the low-budget end (graceful-degradation dominance) — and spans budget points Octree-GS cannot address.

**Exp A2 — Train-under-budget vs prune-after-budget (the moat sharpener, reuses OCB-3DGS-HR).** Overlay a third curve: the render-time post-hoc budget applied to a fully-trained model (the OCB-3DGS-HR M3 method). Claim: training under the budget (Exp A) dominates post-hoc pruning at equal achieved budget — because training redistributes capacity, post-hoc can only subtract. This converts the prior codebase's render-time result into the baseline our training-side method beats, and is a more naturally winnable framing than "beat Octree-GS at matched budget."

**Exp A3 — Unification demonstration (训练-渲染一体化, graduation-core).** Show that one budget governs both ends through the shared anchor population: (i) the budget-shaped training population is exactly what the renderer LoD-activates (no separate render-side budget needed); (ii) report, on that same population, the render-side efficiency (FPS, active-Gaussian count) alongside the training-side quality — establishing that the single `B_total` knob moves training cost, model size, render speed, and quality together. The *stretch* form — `B_v` driving an explicit per-voxel render-time activation quota beyond Octree-GS's distance default, composed with CLoD — is Exp D.

**Exp B — near/far view split (B's targeted value).** Partition test cameras by distance (or by per-view mean depth). Compare A+B vs A-only PSNR on the far/detail subset. Success: A+B ≥ A-only on far views, near views unchanged.

**Exp C — efficiency by-product.** Training time, FPS, active-Gaussian count, and the **A+B vs A-only training-time delta** (B's `2·|camlist|`-render cost must be shown, not hidden). This is the deciding datum for the ADR-0002 Source-B fallback: if B's quality gain does not justify its render cost, B is demoted to a validation-only diagnostic and the system ships A-only.

**Exp D — continuity (stretch).** Reallocation composed with CLoD-GS continuous opacity decay; flicker score / temporal LPIPS vs discrete LoD switching. Out of the graduation-critical path.

**Ablations (from eval-plan §3.3, retained).** Demand source: uniform (= Octree-GS) / gradient-only (`λ=0`) / gradient+photometric (`λ ∈ {0.5, 1.0}`) / photometric-only, holding the entire downstream pipeline identical (fairness condition). Controller knobs: `k_cap`, `θ_frac`, `r%`, `τ_smooth`, `k`. Reallocation headroom: `ρ_min`.

## 5. Success criteria (the bar, made explicit)

Aligned to the proposal's maintain-quality framing, not to beating a baseline:

- **Primary (capability + unification):** a coherent **training-side** Pareto curve exists and is controllable via `B_total`; at a chosen operating point quality is within **≤ 0.5 dB PSNR** of full Octree-GS while active Gaussians are materially lower; and the single budget is shown to move training cost / model size / FPS / quality together on one shared anchor population (Exp A3). Load-bearing graduation result; depends only on the controller running and the sweep + efficiency measurement completing — not on any comparative win.
- **Secondary:** demand curve ≥ uniform curve, with the bar at the **low-budget end** (graceful-degradation dominance — demand keeps far/detail alive where uniform collapses). Moderate effect expected.
- **Secondary′:** training-under-budget curve (Exp A) ≥ render-time post-hoc curve (Exp A2) at equal achieved budget. High prior probability (training redistributes; post-hoc only subtracts).
- **Tertiary:** A+B ≥ A-only on far views.
- A failure of Secondary / Secondary′ / Tertiary does **not** sink the paper: the Primary capability + unification + efficiency + the honest B-leverage characterisation still constitute a defensible contribution (and trigger the ADR-0002 A-only fallback for B).

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
- 🟡 **Rendering-side build cost** (Exp D continuity, and the explicit per-voxel render quota). Mitigated: both are stretch/out-of-critical-path; the graduation core (Exp A/A2/A3/B/C) rides on existing OcbGS code + Octree-GS's native LoD renderer.
- 🟢 **rome "scaling with complexity" over-claim** — guarded in §2; not to be headlined without a tie-corrected re-measure.

## 8. Scope

- **In (graduation-critical):** Exp A (training-side Pareto), **Exp A2 (train-under-budget vs post-hoc, reusing OCB-3DGS-HR as baseline)**, **Exp A3 (structural train/render unification — one budget → one anchor population → LoD render)**, Exp B (view split), Exp C (efficiency + B-cost), the demand-source and controller-knob ablations. Built largely on existing OcbGS code (controller, Source B, `B_total` knob, `exp2_*_pareto.sh` and `collect_results.py` drafts) plus Octree-GS's native LoD renderer for the render-side numbers.
- **Stretch (non-critical):** Exp D continuity (CLoD composition + flicker); the *explicit* unification form (`B_v`-driven per-voxel render-time activation quota beyond Octree-GS distance default); optional undershoot fix to place curve points at chosen budgets.
- **Out:** the full 6-D voxel state vector, the five-state lifecycle machine, and large-scale (MatrixCity-class) scenes. Note: the 6-D state and state machine are **already built in OCB-3DGS-HR**, but for a *render-time* prototype on a different (gsplat) stack — porting them into the OcbGS training side is a fresh integration, so they stay out of the graduation core and are revisited only if the core lands early. (OcbGS's training-side demand uses A = gradient + B = photometric; the other four state dimensions are enhancements, not prerequisites.)
