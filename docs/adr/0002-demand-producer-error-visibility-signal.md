# ADR-0002: Demand Producer — error×visibility signal

**Status:** Accepted
**Source spec:** `docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`

## Context

The demand signal is the starting point of the closed loop (ADR-0001). It must answer the question "which anchor lacks detail?" in a form the rest of the pipeline can consume. Octree-GS already maintains per-iteration accumulators in `training_statis` (`anchor_demon`, `offset_gradient_accum`, `offset_denom`) — the demand signal reuses them at near-zero added cost.

The DemandProducer is partition-agnostic (ADR-0001, constraint 3): it knows nothing of Control Cells, `control_level`, or the Capacity-Budget-derived partition. This ensures the future `SemanticDemand` drops in with zero skeleton change.

## Decision

### D4 — Gradient accumulator as primary (A), FastGS photometric residual as refinement (B)

The primary demand source is a gradient-accumulator-based proxy (Source A), available every iteration for free. A periodic photometric residual from FastGS (Source B) refines it, correcting cases where the gradient proxy is blind. Both are per-anchor, raw, partition-agnostic signals.

### Anchor Demand contract

`s(a) ∈ [0, +∞)` — unitless, non-negative, comparable, raw (no normalisation). No knowledge of Control Cells or `control_level`. The single L1 normalisation and any signal fusion happen downstream in the Controller (ADR-0004).

### Signal architecture: A and B are independent signals sharing one contract

`ErrorVisibilityDemand.produce()` returns a single tensor `s(a)` = Source A (gradient error × visibility). This satisfies the pluggable `DemandProducer` interface contract (§4.5): one per-anchor signal, partition-agnostic.

Source B (FastGS `pruning_score`) is a second, independent per-anchor signal that conforms to the **same contract** — per-anchor, raw, partition-agnostic. It may be produced by the same `ErrorVisibilityDemand` instance via a second method, or by a separate producer instance; both are implementation choices that do not affect the boundary. The architecture decision is: **B is an independent signal, reduced by the same Partition (ADR-0003) independently to `d_B(v)`, and fused with A only at the Controller (ADR-0004).**

### Source A — gradient-accumulator-based (primary, free per iteration)

```
s(a) = error(a) × visibility(a)
```

- **`visibility(a)` = `anchor_demon`** — the raw per-anchor count of views that observed the anchor (`gaussian_model.py:643`). Kept **raw, not** normalised to `[0,1]`: the observation count up-weights anchors whose error affects many views (the FastGS `photometric_loss × accum_loss_counts` philosophy). Dividing by `max(anchor_demon)` would discard exactly that multi-view weight.

- **`error(a)` = masked-max over offsets:**
  ```
  g_k(a)    = offset_gradient_accum[a,k] / offset_denom[a,k]              # per-offset mean grad
  mature(k) = offset_denom[a,k] > check_interval · success_threshold · 0.5  # native offset_mask, :857
  error(a)  = max { g_k(a) : mature(k) }   (0 if a has no mature offset)
  ```

  - **Why max, not mean or sum — grounded in the actuator.** Native `anchor_growing` spawns a candidate per offset whose mean gradient crosses the level threshold (`candidate_mask = grads ≥ cur_threshold`, `:734`) — any single high-gradient offset is already a grow trigger. Max-over-offsets makes the demand signal agree with how growth physically fires; mean would mask a single steep offset (the exact case Octree-GS grows on); sum would bias toward high-`n_offsets` anchors and is not cross-comparable. (The code's own per-anchor reduction `anchor_grads`, `:720`, is a mean — but it serves only the orthogonal extra-level promotion, not the primary grow; demand follows the primary-grow max semantics.)

  - **Why the maturity mask, not just a divide-by-zero guard.** Max is outlier-sensitive: a single 1-observation offset with a noisy large gradient would dominate. The native `offset_mask` (`:857`) already gates growth on sufficiently-observed offsets; reusing it filters the same immature offsets out of the max.

**Caveat — A is a "should-I-densify" proxy, not photometric error, and carries a screen-space scale bias.** `grad_norm = ‖viewspace_point.grad[:2]‖` (`:652`) is a screen-space gradient, and Octree-GS deliberately raises the grow threshold at finer levels (`cur_threshold = threshold · (fork^update_ratio)^cur_level`, `:730`). Both make A under-weight genuinely under-fit fine / distant regions whose screen footprint (hence screen gradient) is small. **We do not hand-correct this inside A** (a per-level rescale would bake octree geometry into the partition-agnostic producer, violating ADR-0001 constraint 3). This bias is precisely the blindness Source B exists to cover.

### Source B — FastGS photometric residual (periodic refinement)

FastGS `compute_gaussian_score_fastgs` produces `pruning_score = photometric_loss × accum_loss_counts` — true photometric error × observation count over a sampled camera set. Used only at the periodic Controller step (every N iterations) to correct the gradient-based demand.

- **Cost of B (must be reported in experiments).** Per B evaluation, two forward renders per camera in the `camlist` (forward-only, no backprop): one for photometric loss, one to accumulate per-Gaussian high-error counts. Cost ≈ `2·|camlist|` forward renders per evaluation. Two knobs bound it: the camera subsample `|camlist|` and the B-period `M` (Controller steps between B evaluations; `d_B` is held between refreshes). Conservative defaults (small subsample, periodic) keep it a few-percent overhead.

- **Why B — covering A's blind spot.** A measures a screen-space gradient proxy; it systematically under-weights fine / distant regions. B is a true photometric residual, independent of screen-space scale, so it can light up exactly the regions A misses. This is the concrete, load-bearing motivation for a second signal — and for additive (not multiplicative) fusion.

### Fusion principle — additive, not multiplicative

The actual fusion operator (L1 normalisation, λ weighting, additive combination, EMA smoothing) lives in the Controller (ADR-0004). The architecture decision that belongs here is **why additive**:

Multiplicative fusion `d_A · (1 + α · d_B)` gates B by A: where the gradient proxy is blind (`d_A ≈ 0`), B cannot raise demand — defeating the correction B exists to provide. Additive fusion lets B independently light up a Control Cell A missed. This principle is a producer-level motivation for B's existence, independent of how the Controller implements the combination.

`λ = 0` recovers A-only. `λ` is a Controller fusion knob (ADR-0004) exposed here only as an ablation axis.

### Fallback — data-driven, not pre-committed

If the ablation shows B's quality gain does not justify its render cost, B is demoted to a **validation-only diagnostic** (check that A's reallocation correlates with true photometric error) and the system ships A-only. The fallback is decided by data (Exp 4: A+B-vs-A training-time delta), not pre-committed in architecture.

### Pluggable interface

```python
class DemandProducer:
    # Contract: return raw, non-negative, comparable per-anchor Anchor Demand s(a)
    # in [0, +∞), one per anchor. No units, no normalisation, no knowledge of
    # Control Cells / control_level (partition-agnostic — ADR-0001 constraint 3).
    # The s(a) → d(v) reduction lives in Partition (ADR-0003); the single L1
    # normalisation lives in Controller (ADR-0004).
    def produce(self, scene, stats) -> Tensor:  # shape [N_anchors] = s(a)
        ...

# now:    ErrorVisibilityDemand — reads training_statis (+ FastGS)
# future: SemanticDemand (semantic mask → per-anchor score, same contract)
```

## Consequences

- Each producer emits only a per-anchor raw signal; it never emits `d(v)`. The `s(a) → d(v)` reduction belongs to Partition (ADR-0003). Signal fusion and normalisation belong to the Controller (ADR-0004).
- `s(a)` is computed once and fans out to two consumers: Partition's reduction (`s(a) → d(v)`) and the Actuator's prune ranking (lowest-`s(a)` per surplus Control Cell).
- Source B's render cost must be explicitly reported in the evaluation (Exp 4: A+B vs A-only training-time delta). It is the deciding data point for the fallback.
- Ablation axes: `λ` (fusion weight, home in ADR-0004), `|camlist|` (camera subsample for B), `M` (B refresh period in Controller steps).

## Non-goals

- Per-anchor normalisation of `s(a)`. The single L1 normalisation lives in the Controller (ADR-0004).
- Per-Control-Cell aggregation `s(a) → d(v)`. This is the Partition's responsibility (ADR-0003).
- The fusion operator (normalise, λ-weight, add, EMA). The principle of additive fusion is decided here; the mechanism lives in the Controller (ADR-0004).
- Budget normalisation `d(v) / Σd`. This is the Controller's responsibility (ADR-0004).
- Implementation of `SemanticDemand`. The interface is reserved; the implementation is future work.
