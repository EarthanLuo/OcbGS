# ADR-0003: Partition & control_level derivation

**Status:** Accepted
**Source spec:** `docs/superpowers/specs/2026-06-19-demand-driven-budget-reallocation-design.md`

## Context

The DemandProducer (ADR-0002) emits per-anchor Anchor Demand `s(a)`. The Budget Controller (ADR-0004) requires per-Control-Cell Demand Field `d(v)` and Cell Occupancy `n(v)`. The Partition is the pure reduction bridge between them — it produces no signals, moves no anchors, and runs no CUDA. Its sole job is the segment-sum `s(a) → d(v)` over Control Cells.

The spatial granularity at which the Capacity Budget is allocated must also be determined. Too fine a partition (one anchor per Control Cell) leaves no room to reallocate; too coarse (one Control Cell total) leaves nothing to reallocate between. The `control_level` — the octree level at which Control Cells are formed — is derivable from the Capacity Budget and a minimum mean occupancy, not a free knob.

The Partition is the octree-geometry-aware side of the architecture, but the DemandProducer remains partition-agnostic (ADR-0001, constraint 3). When the future `SemanticDemand` replaces `ErrorVisibilityDemand`, the Partition is reused verbatim.

## Decision

### Control Cell — the partition unit

A Control Cell is an occupied octree cell at the `control_level`. The defining rule: a box at this level becomes a Control Cell once it holds an anchor or carries demand; empty space is not a Control Cell. Control Cells at a single level form a non-overlapping spatial partition, required for Capacity Budget normalisation and the Budget Constraint.

Octree-GS samples anchors at every level `cur_level ∈ [0, levels)`; anchors at all levels coexist, and a fine cell is one of the `fork³` children of its parent coarse cell. The `control_level` selects exactly one of these levels as the partition grid. Capacity within a Control Cell is realized at finer levels: reallocating capacity from low-demand to high-demand Control Cells means low-demand cells stop growing deeper or get their fine-level anchors pruned (coarsen), while high-demand cells grow deeper (subdivide). The execution of grow/prune/subdivide belongs to the Actuator (ADR-0005); the Partition only defines the spatial bins.

### Cell Membership — stateless, by position, `round` not `floor`

Each anchor belongs to exactly one Control Cell: the cell whose box contains the anchor's centre position, **independent of the anchor's own octree level**. A coarse anchor is billed to the single cell containing its centre — no area-weighted splitting across cells it spatially spans, since fractional anchors would break both the partition (non-disjoint cells) and the grow/prune actuator.

Membership is **stateless** — recomputed each Controller step as `cell_id = round((anchor_pos − init_pos) / cell_size)` (a vectorised round-division, O(N_anchors), pure tensor, millisecond-scale at 10⁶ anchors). No incremental `anchor → cell` dictionary is maintained. Statelessness is strictly correct whether or not anchor positions move: an incremental map would be invalidated by any boundary crossing. Octree-GS freezes anchor positions (`position_lr = 0`, asserted in config — only `_offset` moves), so an anchor's accumulated `s(a)` over a Controller window always belongs to the same cell (no demand smearing across boundaries).

**`round`, not `floor`.** The native octree grid snap in Octree-GS (`gaussian_model.py:752/754`) uses round-division to assign anchors to grid cells. We match that convention so a Control Cell equals an octree cell at the `control_level` and a fine anchor maps correctly to its true ancestor cell at that level. `floor` would shift the grid by half a cell and mis-assign boundary anchors (half of them would map to the wrong ancestor).

### `control_level` — derived, not a free knob

The feasible range of `control_level` is bounded at both ends by the active-cell count `N_active(level)` (number of occupied Control Cells at that level) and the mean occupancy `m(level) = B_total / N_active(level)` (average anchors per Control Cell — the per-cell reallocation headroom):

- **Too fine** ⇒ `N_active → B_total` ⇒ `m → 1` ⇒ nearly every Control Cell sits at the floor with no room to grow or shrink ⇒ the Controller degenerates to identity.
- **Too coarse** ⇒ `N_active → 1` ⇒ nothing to reallocate between ⇒ degenerates to uniform.

The control knob is **mean occupancy**, not an aggregate budget fraction. Given a **minimum mean occupancy `ρ_min`** (each Control Cell averages at least `ρ_min` anchors, so it has room to move) and a minimum cell count `A_min`, `control_level` is derived:

```
control_level = max { level :
    B_total / N_active(level) ≥ ρ_min
    ∧ N_active(level) ≥ A_min
}
```

i.e. the finest level whose cells still average `≥ ρ_min` anchors and number `≥ A_min`. Fix `ρ_min`, and `control_level` falls out of `B_total` automatically. The ablation sweeps `ρ_min` (not raw `control_level`).

**Why `ρ_min`, not the earlier aggregate-headroom fraction `τ` (rejected alternative).** Under `floor = 1` the two are exactly linked — `m = 1 / (1 − τ)` — but `τ` is mis-conditioned as a knob: a conservative-sounding `τ = 0.3` (30% of the Capacity Budget kept free) corresponds to `m = 1 / 0.7 ≈ 1.43` anchors per Control Cell, i.e. the floor-pinned degenerate regime. The aggregate "30% free" hides that the free budget is spread so thin that per Control Cell there is essentially no headroom. `ρ_min` states the per-cell headroom directly (`ρ_min ≈ 8 ⇔ τ ≈ 0.875`) and is the well-conditioned form of the same constraint.

**Safety property.** `ρ_min > floor` (always true for defaults `floor = 1`, `ρ_min = 8`) implies `floor · N_active < B_total`, so the Budget Constraint is always physically satisfiable at the derived `control_level`. Although `floor` is a Controller knob (ADR-0004), this safety guarantee is a property of the derivation in this module.

Defaults: `ρ_min = 8` (8 anchors per Control Cell on average; floor eats only `floor / m ≈ 1 / ρ_min ≈ 12%`), `A_min = 10`.

**`control_level` is derived once at Controller activation, then frozen.** It is not derived at Partition construction (a pre-unlock snapshot of anchor positions under-populates fine levels, producing a spuriously coarse level). It is never re-derived from a count — `update_control_level(N_total)` is type-wrong (derivation needs positions to compute `N_active(level)`) and would thrash the `d`/`n` bookkeeping mid-training. The level is forward-looking: it uses `B_total` (not the current anchor count), so the Phase-1 ramp never needs a re-derive.

### Partition API contract

```python
class Partition:
    """Owns control_level + Cell Membership + the s(a) → d(v) reduction.
    Stateless except for the once-derived, frozen control_level (hence cell_size);
    membership is recomputed each call (no incremental anchor→cell map)."""

    def __init__(self, B_total: int, floor: int, rho_min: float, A_min: int,
                 voxel_size: float, fork: int, levels: int, init_pos: Tensor):
        ...  # stores config only; control_level is NOT derived here.

    def set_control_level(self, anchor_positions: Tensor) -> int:
        ...  # Called ONCE at Controller activation (post full unlock, ADR-0004).
             # Derives control_level from anchor_positions and freezes
             # cell_size = voxel_size / fork**control_level. Forward-looking:
             # uses B_total (not the current count), so the Phase-1 ramp needs
             # no re-derive.
             # The call is a one-time guard in adjust_anchor (first post-unlock
             # step), not repeated every Controller step — an implementation
             # detail not shown in the ADR-0005 per-step orchestrator pseudocode.

    def cell_id(self, anchor_positions: Tensor) -> Tensor:        # int64 [N]
        ...  # round((pos - init_pos) / cell_size) flattened to an integer id.
             # ROUND (not floor), matching the native octree grid
             # (gaussian_model.py:752/754) so a Control Cell == an octree cell
             # at control_level. Pure positional ⇒ valid on not-yet-inserted
             # grow candidates. PUBLIC: the Actuator (ADR-0005) calls it to bin
             # candidates for the per-Control-Cell cap.

    def reduce(self, anchor_positions: Tensor, weights: Tensor,
               exclude: Tensor | None = None) -> tuple[Tensor, Tensor]:
        ...  # segment-sum over Cell Membership →
             # (active_cell_ids[N_active], values[N_active]).
             # d(v) = reduce(pos, s_a);
             # n(v) = occupancy = reduce(pos, ones, exclude) — reduce with
             # unit weights, not a separate method.
             # exclude is a pure input mask (dead-anchor GC set, ADR-0005);
             # Partition mutates nothing and computes post-GC d/n analytically.
```

Key design consequences of this API:

- **Stable `cell_id`s, not a dense `[N_cells]` vector.** The active Control Cell set changes as anchors grow and are pruned, so a dense vector's index `i` would denote different cells across Controller steps. `reduce` returns `(active_cell_ids, values)` so the Controller can align cells across steps — required by the Spearman gate ("over their shared cells", ADR-0004) and by the Actuator's `cell_id → plan-δ` lookup (ADR-0005). The `ReallocationPlan` is keyed by these same ids (ADR-0004).
- **`reduce`/membership take positions as input** (not cached). Partition holds no per-anchor state, only the frozen `control_level`/`cell_size`, preserving statelessness.
- **`exclude` is a pure input mask.** The dead-anchor GC set (ADR-0005, Step 0) is generated by the Actuator and passed in. Partition computes post-GC `d` and `n` analytically without mutating any state.
- **The cap stays in the Actuator, not Partition.** Partition exposes `cell_id`; the Actuator ranks each Control Cell's candidates by proposing gradient and keeps the top `δ⁺(v)`. A `cap_by_cell` on Partition would pull gradient-ranking + candidate materialisation into the membership module, breaking its single responsibility.

### Per-Control-Cell demand

`d(v) = Σ_{a : member(a)=v} s(a)` — a pure segment-sum over Cell Membership. `d(v)` is the **Demand Field**: the collection of Demand Scores over all Control Cells, reduced from Anchor Demand. Unitless, non-negative, cross-cell-comparable; relative and rank-meaningful only. The Controller's single L1 normalisation turns it into an allocation weight (ADR-0004).

Cell Occupancy `n(v)` = number of anchors with Cell Membership in `v` — `reduce(pos, ones, exclude)`.

## Consequences

- Partition is **pure CUDA-free**: membership is a vectorised round-division (O(N_anchors)), reduction is a segment-sum (O(N_anchors)). Both are millisecond-scale at million-anchor counts, locally unit-testable on Windows.
- Stable `cell_id`s are the cross-step alignment primitive the Controller (Spearman gate) and Actuator (plan lookup) depend on. Every consumer of the Partition output keys its data by these ids.
- A grow candidate in a previously-empty (unplanned) Control Cell receives zero quota in the controlled phase: the `ReallocationPlan` covers only cells active at planning time, and admitting unbudgeted new-cell growth would break `Σn ≤ B_total` (no force-fill, ADR-0004).
- `control_level` derivation is reproducible and deterministic given `B_total`, `ρ_min`, `A_min`, and the activation-time anchor positions.
- Ablation axis: sweep `ρ_min ∈ {4, 8, 16}` — granularity vs per-Control-Cell-headroom trade-off.

## Non-goals

- Budget normalisation `d(v) / Σd`, clamping `[floor, cap]`, water-fill, integer apportionment — these are the Controller's responsibility (ADR-0004).
- The Spearman rank-stability gate — Controller logic (ADR-0004).
- Dead-anchor GC mask generation — Actuator responsibility (ADR-0005). Partition receives the `exclude` mask as a pure input.
- Per-Control-Cell candidate ranking by proposing gradient and selecting top `δ⁺` — Actuator responsibility (ADR-0005). Partition only exposes `cell_id` for binning.
- Execution of grow/prune/subdivide on anchors — Actuator responsibility (ADR-0005).
