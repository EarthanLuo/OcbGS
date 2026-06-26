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

### Update 2026-06-27 — actuator-relax experiment: B demoted, controllable budget promoted

The 73% budget undershoot was traced in code to the actuator only ever *filtering* native gradient-spawned candidates (`anchor_growing_capped` → `_anchor_growing_gather` gates candidate creation on `grads ≥ cur_threshold`, gaussian_model.py:888), so the demand cells B points at (low gradient → no candidates) could never grow — B changed the *plan* but the actuator dropped B's distinctive part. A `--grow_relax_scale` knob (scales that threshold on the controller growth path only; default 1.0 = unchanged) was added to test "does demand allocation help once the controller can actually move?". Two findings on amsterdam (seed 0):

- **Controllability unlocked → promote.** With `--grow_relax_scale 0.1` the controller fills **99.7%** of `B_total` (1,100,846 / 1,104,448), up from ~73%. The set-point is now *hittable*, so the controllable-budget Pareto mainline is feasible and the §3 undershoot is **resolved**, not merely sidestepped.
- **B refuted as a quality / far-view lever → demote.** A+B vs A-only, both relaxed, matched budget: overall PSNR is a **tie** (28.398 vs 28.332; +0.066 dB = noise). A near/far view split (`eval_view_split.py`) shows B's entire tiny edge comes from **near** views (+0.141) while **far** views — B's §4.1 home turf — are flat-to-worse (**−0.017**). That is the *opposite* of the Tertiary thesis, and it holds on the **train** set (B's best case, where it actually placed the capacity). With the probe (B only ~60% rank-independent, moderate) and the overall tie, three independent signals converge: **B is not a quality lever.** Per the §4.1 fallback + ADR-0002, B is **demoted to a validation-only diagnostic / documented negative result; the system ships A-only.** Held-out-test confirmation (rendered from the checkpoints, no retrain) **strengthens** it: far ΔPSNR = **−0.218** on test (worse than train's −0.017), overall net −0.026. Remaining confidence caveat: `n = 1` scene (amsterdam)/seed; multi-scene would harden it further but is not needed to act.

Net effect on the tiers below: **Primary and the efficiency story carry the paper**; the relax work feeds that surviving mainline. The B-dependent **Tertiary is retired to a negative result**, and the **Secondary** demand-vs-uniform claim is now an *open, A-only* question (does gradient-demand allocation beat uniform at low budgets?), no longer relying on B.

## 2. Claims (tiered by load-bearing importance)

**Primary — controllable, training-side budget with a unified train/render decision (capability / moat, not a superiority claim).** A single Capacity Budget governs the *training-time* anchor population — demand-driven reallocation across Control Cells under `B_total` — and that same LOD-structured population is what the renderer activates per view. The budget therefore drives **both ends through the shared octree anchor structure**: training shapes which anchors exist (and at which level), rendering activates a distance-subset of exactly that population. This closes the proposal's "information gap" (训练-渲染一体化) *structurally* — one demand-shaped, budget-bounded anchor population serves both. Sweeping `B_total` traces a **training-side** quality–cost Pareto front (the model is *built for* each budget, not pruned after). The novelty is the combination — octree LOD structure **+** an explicit budget set-point **+** demand-driven spatial distribution during training — which no single prior method offers: 3DGS-MCMC has a global count cap but no LOD structure and no spatial demand allocation; Octree-GS has LOD but no budget set-point; CDC-GS / Taming target other axes; and post-hoc render-time budgeting (OCB-3DGS-HR M3) only degrades a fixed model.

**Secondary — gradient-demand allocation dominates uniform, especially at low budgets (graceful-degradation dominance). OPEN.** At matched *achieved* budget, A-only (gradient) demand allocation should lie on or above the uniform-allocation curve (uniform = Octree-GS's own spatial distribution), most defensibly **at the low-budget end**: as the budget shrinks, demand keeps capacity where reconstruction error is while uniform lets it collapse — so our curve stays up where uniform falls off. This is "better where it matters" framed as *graceful degradation*, not beating a full baseline. It is a *relative* claim against our own uniform control; it only needs to hold, not dominate everywhere. **Status: untested at low budgets.** Note the matched-budget tie at 1× (§update) says nothing about the low-budget end — at 1× there is little to reallocate; leverage is expected to grow as the budget tightens. If this also turns out flat, the paper still stands on Primary (controllable budget + unification) + efficiency.

**Tertiary — RETIRED to a negative result (was: B is a targeted corrector of A's blind spot).** The hypothesis was that Source B surfaces under-fit fine/distant regions A's screen-space gradient under-weights (2026-06-19 spec §4.1), and would beat A-only on a far-view split. **Tested and refuted** (see the 2026-06-27 update above): at matched budget A+B ties A-only overall (+0.066 dB) and is flat-to-worse on the far views it was meant to help (−0.017), with its only gain on near views — the opposite of the thesis. B is demoted to a validation-only diagnostic (ADR-0002) and reported as an honest negative result; the system ships **A-only**. The paper does not rely on this claim.

**Efficiency reporting (rides on existing machinery).** FPS, active (opacity-masked) Gaussian count, training time — largely inherited from the Octree-GS LoD renderer; measured and reported against the maintain-quality bar, not engineered anew.

**Continuity (stretch).** CLoD-GS continuous opacity decay composed with reallocation, evaluated by flicker score / temporal LPIPS. This is the only tier that requires *new* rendering-side code beyond Octree-GS's native LoD renderer; it is explicitly out of the graduation-critical path.

### Probe evidence (characterises B; was the basis for the now-retired Tertiary)

A per-Control-Cell instrumentation of `d_A` and `d_B` at the controller's real operating window (BungeeNeRF, iters 7000–11000, converged regime) established:

- **B is not flat.** Per-cell `d_B` is heavy-tailed: coefficient of variation ≈ 1.27, p99/p50 ≈ 9.8, ~98% of cells non-zero (amsterdam; all three scenes show CV > 1).
- **B is ~60% rank-independent of A.** Spearman(`d_A`, `d_B`) ≈ 0.58–0.69 across amsterdam/quebec/rome (R² ≈ 0.41 → ~59% of B's rank variance is unexplained by A); top-decile overlap ≈ 0.50 (half of A's highest-demand cells are *not* in B's top decile). Stable across the window — structure, not noise.

This refutes the earlier worry that B might be degenerate, and sizes the leverage as **moderate, not dramatic** — which is why the mainline does not stake the paper on B beating anything.

*Caveat carried into the spec:* the "B is more independent in larger scenes" reading (rome 185k cells, Spearman 0.577 lowest) is **not** yet a defensible trend — `n = 3`, and the naive `argsort` rank used in the probe is not tie-corrected, so more zero-`d_A` cells in large scenes can mechanically depress the correlation. Treat as suggestive; do not headline without a tie-corrected re-measure on more scenes.

## 3. The 73% undershoot — now resolved

Two independent reasons it is no longer a blocker:

1. **Plotting on achieved budget never needed it.** The Pareto curve is plotted on *achieved* budget; a 73%-of-set-point run is simply a point further left on the x-axis. The matched-budget claim was the only consumer of "hit `B_total` exactly"; with it retired, no force-fill is required and the design stays consistent with ADR-0003/0004/0005.
2. **It is now actually fixed (2026-06-27).** `--grow_relax_scale 0.1` lets the controller's growth path spawn candidates below the native gradient threshold, so the controller fills **99.7%** of `B_total`. The set-point is hittable, which upgrades the Primary claim from "traces *a* curve" to "traces a *controllable* curve at chosen budgets." The relax knob is an experimental switch today; promoting it into the method (a principled candidate-relaxation in demand cells, distinct from the retired force-fill-to-match-budget) is a graduation-core cleanup, since hitting the set-point *is* the controllability capability.

## 4. Experiments

Baselines (from eval-plan §1, unchanged): Vanilla 3DGS (capacity-agnostic reference), **Octree-GS = primary control = uniform allocation**, CLoD-GS (continuous-LOD comparison), FastGS (training-speed axis, optional). Octree-GS emits one point; ours sweeps.

**Exp A — Training-side Pareto front (the money figure).** Per scene × seeds, sweep `B_total ∈ {0.25, 0.5, 1, 2} ×` the per-scene Octree-GS baseline anchor count (measurement protocol = eval-plan §4, unchanged). **Each point is a full budget-guided training run** — the model is *built for* that budget, not pruned afterwards. Record *achieved* active anchors / opacity-masked Gaussians (x) vs PSNR/SSIM/LPIPS (y). Two curves: **demand (A+B)** and **uniform control**. Success: the demand curve lies on or above uniform across the swept range — most importantly at the low-budget end (graceful-degradation dominance) — and spans budget points Octree-GS cannot address.

**Exp A2 — Train-under-budget vs prune-after-budget (the moat sharpener, reuses OCB-3DGS-HR).** Overlay a third curve: the render-time post-hoc budget applied to a fully-trained model (the OCB-3DGS-HR M3 method). Claim: training under the budget (Exp A) dominates post-hoc pruning at equal achieved budget — because training redistributes capacity, post-hoc can only subtract. This converts the prior codebase's render-time result into the baseline our training-side method beats, and is a more naturally winnable framing than "beat Octree-GS at matched budget."

**Exp A3 — Unification demonstration (训练-渲染一体化, graduation-core).** Show that one budget governs both ends through the shared anchor population: (i) the budget-shaped training population is exactly what the renderer LoD-activates (no separate render-side budget needed); (ii) report, on that same population, the render-side efficiency (FPS, active-Gaussian count) alongside the training-side quality — establishing that the single `B_total` knob moves training cost, model size, render speed, and quality together. The *stretch* form — `B_v` driving an explicit per-voxel render-time activation quota beyond Octree-GS's distance default, composed with CLoD — is Exp D.

**Exp B — near/far view split (B's targeted value). DONE — negative, held-out-test confirmed.** `eval_view_split.py` partitions views by camera distance and reads the existing per-view metrics. amsterdam, seed 0, A+B vs A-only far-view ΔPSNR: **−0.017 (train)** and **−0.218 (held-out test)** — B makes the far views it was meant to help *worse*, more so on test; overall net −0.026 on test, with the only gain near-only (+0.13). A+B also rendered *more* Gaussians (1.81M vs 1.70M) for that net-negative quality. Held-out test was rendered from the saved checkpoints with `--eval --ape -1` + `metrics.py` (no retrain). The negative is now publication-grade. (Appearance caveat: this codebase only ever evaluated the train split; test appearance uses shifted train uid embeddings, so absolute test PSNR is approximate but the arm_b/arm_c comparison is fair — a proper test-time appearance handling is owed for the final paper.) This experiment serves the negative result, not a positive claim.

**Exp C — efficiency by-product.** Training time, FPS, active-Gaussian count for the shipped **A-only** system, reported against the maintain-quality bar. The **A+B vs A-only training-time delta** (B's `2·|camlist|`-render cost) is reported as part of the negative result: B costs extra render time for no quality gain — the ADR-0002 fallback has **fired** (B demoted), so the shipped system is A-only.

**Exp D — continuity (stretch).** Reallocation composed with CLoD-GS continuous opacity decay; flicker score / temporal LPIPS vs discrete LoD switching. Out of the graduation-critical path.

**Ablations (from eval-plan §3.3, retained).** Demand source: uniform (= Octree-GS) / gradient-only (`λ=0`) / gradient+photometric (`λ ∈ {0.5, 1.0}`) / photometric-only, holding the entire downstream pipeline identical (fairness condition). Controller knobs: `k_cap`, `θ_frac`, `r%`, `τ_smooth`, `k`. Reallocation headroom: `ρ_min`.

## 5. Success criteria (the bar, made explicit)

Aligned to the proposal's maintain-quality framing, not to beating a baseline:

- **Primary (capability + unification):** a coherent **training-side** Pareto curve exists and is controllable via `B_total`; at a chosen operating point quality is within **≤ 0.5 dB PSNR** of full Octree-GS while active Gaussians are materially lower; and the single budget is shown to move training cost / model size / FPS / quality together on one shared anchor population (Exp A3). Load-bearing graduation result; depends only on the controller running and the sweep + efficiency measurement completing — not on any comparative win.
- **Secondary (OPEN, A-only):** the gradient-demand curve ≥ uniform curve, bar at the **low-budget end** (graceful-degradation dominance). Untested; the 1× matched-budget tie does not bear on it. Moderate effect hoped, not required.
- **Secondary′:** training-under-budget curve (Exp A) ≥ render-time post-hoc curve (Exp A2) at equal achieved budget. High prior probability (training redistributes; post-hoc only subtracts).
- **~~Tertiary: A+B ≥ A-only on far views.~~ FAILED (2026-06-27):** far-view ΔPSNR = −0.017 on amsterdam; B demoted to diagnostic / negative result, system ships A-only.
- A failure of Secondary / Secondary′ does **not** sink the paper: the Primary capability + unification + efficiency constitute the defensible contribution on their own; the Tertiary failure is already absorbed (B demoted, A-only shipped).

## 6. Relationship to existing docs (consistency map)

- **Retires** the *matched-budget* operating point: 2026-06-19 spec §6.3 line 309; eval-plan §3 line 28; and the "critical comparison = equal #anchors" framing (spec line 298, eval-plan line 14) → re-stated as "comparison along the Pareto curve at matched *achieved* #anchors, demand vs uniform."
- **Preserves unchanged** the no-force-fill invariant and all of ADR-0003/0004/0005 — now fully consistent, since the only clause that required force-fill is gone.
- **Adds** the near/far view-split evaluation (eval-plan §5 currently has only the capacity heatmap).
- **Keeps** `CONTEXT.md` as-is — its "honest slack, not padded with low-value anchors" already encodes the natural-budget stance.
- **Maps to the proposal:** Primary claim = proposal 创新点一 (unified voxel budget) + 主要问题五 (controllable trade-off / Pareto). The proposal needs no change — it already asks for maintain-quality + controllable budget, neither of which depends on B. Source B / the §4.1 screen-space blind-spot correction (proposal's refinement signal) becomes a *tested negative result*, not a headline contribution.
- Old spec is retained (not deleted); a pointer to this document is added at its top.

## 7. Risks & fallbacks

- ✅ **B as a quality lever — RESOLVED (negative).** Tested and refuted (far ΔPSNR −0.017); B demoted to diagnostic, A-only shipped (ADR-0002 fallback fired). No longer a risk, now a documented result. Owed: held-out-test confirmation for paper-grade rigor.
- 🟡 **Gradient-demand may only tie uniform at low budgets** (Secondary). Mitigated: Primary (controllable budget + unification) + efficiency carry the paper without it; tie reported honestly.
- 🟡 **Rendering-side build cost** (Exp D continuity, and the explicit per-voxel render quota). Mitigated: both are stretch/out-of-critical-path; the graduation core (Exp A/A2/A3/C) rides on existing OcbGS code + Octree-GS's native LoD renderer.
- 🟢 **rome "scaling with complexity" over-claim** — guarded in §2; not to be headlined without a tie-corrected re-measure.

## 8. Scope

- **In (graduation-critical):** Exp A (training-side Pareto), **Exp A2 (train-under-budget vs post-hoc, reusing OCB-3DGS-HR as baseline)**, **Exp A3 (structural train/render unification — one budget → one anchor population → LoD render)**, Exp B (view split), Exp C (efficiency + B-cost), the demand-source and controller-knob ablations. Built largely on existing OcbGS code (controller, Source B, `B_total` knob, `exp2_*_pareto.sh` and `collect_results.py` drafts) plus Octree-GS's native LoD renderer for the render-side numbers.
- **Stretch (non-critical):** Exp D continuity (CLoD composition + flicker); the *explicit* unification form (`B_v`-driven per-voxel render-time activation quota beyond Octree-GS distance default); optional undershoot fix to place curve points at chosen budgets.
- **Out:** the full 6-D voxel state vector, the five-state lifecycle machine, and large-scale (MatrixCity-class) scenes. Note: the 6-D state and state machine are **already built in OCB-3DGS-HR**, but for a *render-time* prototype on a different (gsplat) stack — porting them into the OcbGS training side is a fresh integration, so they stay out of the graduation core and are revisited only if the core lands early. (OcbGS's training-side demand uses A = gradient + B = photometric; the other four state dimensions are enhancements, not prerequisites.)
