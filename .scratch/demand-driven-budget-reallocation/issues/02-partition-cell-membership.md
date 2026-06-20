# Issue: `ocbgs/partition/` — Cell Membership + control_level derivation + segment-sum reduction

**Status:** ready-for-agent

## What to build

Implement the `Partition` class in `ocbgs/partition/`. The Partition is a pure CUDA-free bridge: it reduces per-anchor `s(a)` into per-Control-Cell `d(v)` (the Demand Field) and computes Cell Occupancy `n(v)`.

**`Partition` API contract (ADR-0003):**

```python
class Partition:
    def __init__(self, B_total, floor, rho_min, A_min,
                 voxel_size, fork, levels, init_pos):
        ...  # stores config; control_level is NOT derived here

    def set_control_level(self, anchor_positions: Tensor) -> int:
        ...  # Called ONCE at Controller activation (first post-unlock step).
             # Derives control_level from anchor_positions, freezes cell_size.
             # Uses B_total (forward-looking); no per-step re-derive.

    def cell_id(self, anchor_positions: Tensor) -> Tensor:
        ...  # round((pos - init_pos) / cell_size) flattened to int64 id.
             # ROUND (not floor), matching native octree grid snap.

    def reduce(self, anchor_positions, weights, exclude=None) -> (Tensor, Tensor):
        ...  # segment-sum over Cell Membership → (active_cell_ids, values).
             # d(v) = reduce(pos, s_a); n(v) = reduce(pos, ones, exclude=GC_mask).
             # exclude is a pure input mask (dead-anchor GC set); Partition computes
             # post-GC d/n analytically without mutating state.
```

**Cell Membership — stateless, `round`, by position.** `cell_id = round((anchor_pos − init_pos) / cell_size)`. Each anchor belongs to exactly one Control Cell regardless of its own octree level. Membership is recomputed each call (no incremental map).

**`control_level` derivation (ADR-0003 § derivation):**

```
control_level = max { level :
    B_total / N_active(level) ≥ ρ_min
    ∧ N_active(level) ≥ A_min
}
```

i.e. the finest level whose cells average ≥ ρ_min anchors and number ≥ A_min. Derived once at activation, frozen. `cell_size = voxel_size / fork**control_level`.

**Reduce with `exclude` mask.** When `exclude` is provided, the masked anchors are excluded from both the count and the sum. This is the post-GC occupancy path used by the Controller (ADR-0004 Step 0).

**Stable `cell_id`s, not a dense vector.** The active Control Cell set changes across training steps. `reduce` returns `(active_cell_ids, values)` so the Controller can align cells across steps (required by Spearman gate and plan lookup).

## Acceptance criteria

- [ ] `cell_id()`: a batch of known positions in a regular grid maps to correct integer cell ids (round semantics verified)
- [ ] `cell_id()`: anchor at cell boundary `.5` rounds away from origin (round, not floor — verify against native `gaussian_model.py:752/754`)
- [ ] `reduce()`: unit weights over known positions → correct per-cell occupancy counts
- [ ] `reduce()`: weighted values (synthetic `s(a)`) → correct per-cell segment-sums
- [ ] `reduce()` with `exclude` mask: excluded anchors contribute zero to both count and sum
- [ ] `set_control_level()`: coarse positions (few cells) → fine level; fine positions (many cells) → coarse level; obeys ρ_min and A_min constraints
- [ ] `set_control_level()`: called once at activation, cell_size frozen thereafter; calling it again is a no-op or raises
- [ ] Safety property: derived `control_level` satisfies `floor · N_active < B_total` (verifiable in test)
- [ ] No CUDA import in `ocbgs/partition/` (local-testable invariant)

## Blocked by

- 00-walking-skeleton (package layout + Partition ABC stub)
