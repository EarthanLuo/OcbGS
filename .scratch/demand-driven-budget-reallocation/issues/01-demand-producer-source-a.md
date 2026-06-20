# Issue: `ocbgs/demand/` — ErrorVisibilityDemand Source A

**Status:** DONE

## What to build

Implement the `DemandProducer` ABC and `ErrorVisibilityDemand` class in `ocbgs/demand/`. Source A emits per-anchor Anchor Demand `s(a)` from Octree-GS `training_statis` accumulators at near-zero added cost.

**`DemandProducer` ABC contract (ADR-0002 § interface):**

```python
class DemandProducer(ABC):
    def produce(self, scene, stats) -> Tensor:  # shape [N_anchors], [0, +∞)
        ...
```

Partition-agnostic: knows nothing of Control Cells, `control_level`, or the Capacity Budget.

**`ErrorVisibilityDemand.produce()` — Source A: `s(a) = error(a) × visibility(a)`**

- **`visibility(a)` = `anchor_demon`** — raw per-anchor view count, NOT normalised to [0,1]. Multi-view weight is preserved.
- **`error(a)` = masked-max over offsets:**
  1. `g_k(a) = offset_gradient_accum[a,k] / offset_denom[a,k]` (per-offset mean gradient)
  2. `mature(k) = offset_denom[a,k] > check_interval · success_threshold · 0.5` (reuse native `offset_mask` maturity gate)
  3. `error(a) = max { g_k(a) : mature(k) }` (0 if no mature offset). Max, not mean — matches native `anchor_growing` trigger semantics.
- Multiply: `s(a) = error(a) × visibility(a)`.

**Caveat (documented, not corrected):** Source A is a screen-space gradient proxy that under-weights fine/distant regions. This bias is the blind spot Source B (issue 06) exists to cover. No per-level rescale is applied (would bake octree geometry into the partition-agnostic producer).

**`_opacity_dead_mask` does NOT live here** — that is the Actuator's responsibility (issue 04).

## Acceptance criteria

- [x] `DemandProducer` ABC is defined with the documented contract signature
- [x] `ErrorVisibilityDemand.produce()` returns a tensor of shape `[N_anchors]`, all values ≥ 0
- [x] With synthetic `training_statis` accumulators (known `anchor_demon`, `offset_gradient_accum`, `offset_denom`), the output matches hand-computed `error × visibility`
- [x] Masked-max: anchor with one mature offset at gradient 0.5 and two immature at 99 → error = 0.5 (immature filtered)
- [x] Anchor with all-immature offsets → error = 0 → `s(a) = 0`
- [x] Anchor with zero `anchor_demon` → `s(a) = 0`
- [x] No import of `gaussian_model` or any CUDA module in `ocbgs/demand/` (local-testable invariant)
- [x] `ErrorVisibilityDemand.produce()` takes `scene` and `stats` — it does NOT reach into model internals beyond what is passed

## Blocked by

- 00-walking-skeleton (package layout + demand ABC stub)
