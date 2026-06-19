# OcbGS

Demand-driven, budget-conserving reallocation of Gaussian capacity for
LOD-structured 3D Gaussian Splatting. This glossary fixes the project's language;
it contains no implementation details.

## Scene Structure

**Anchor**:
A node in the octree holding a small set of offset Gaussians; the unit that is
grown, pruned, and counted toward the budget. Belongs to exactly one octree level.
_Avoid_: point, splat (those are the rendered Gaussians, not the anchor)

**Octree Level**:
One resolution tier of the octree. Anchors coexist across all levels; a finer
level's cell is one of the `fork³` children of its parent. Rendering activates a
subset of levels per view by camera distance.
_Avoid_: LOD tier, depth

**Control Cell**:
A non-overlapping spatial partition unit — an *occupied* octree cell at the
`control_level` (a box becomes a Control Cell once it holds an anchor or carries
demand; empty space is not a Control Cell). The granularity at which demand is
bucketed and capacity is allocated. A pure partition unit; it is not rendered and
carries no opacity/color.
_Avoid_: voxel, cell (bare); "voxel" in older notes means Control Cell

**Control Level**:
The octree level at which Control Cells are formed — i.e. the spatial granularity
of Capacity Budget allocation. A derived quantity, not a free knob: chosen from the
Capacity Budget and a minimum mean occupancy (the finest level whose Control Cells
still hold enough anchors each to have room to move).
_Avoid_: control resolution, allocation level

**Cell Membership**:
The rule assigning each anchor to exactly one Control Cell — the cell whose box
contains the anchor's position, independent of the anchor's own level. Partitions
the entire anchor set into disjoint cells (so the Budget Constraint covers every
anchor exactly once). A coarse anchor is assigned by its center to one cell, not
split across the cells it spatially spans.
_Avoid_: ancestral aggregate, column

## Capacity & Budget

**Capacity Budget** (`B_total`):
The fixed integer upper bound on the total number of anchors the system may hold.
A resource quantity, not a rule.
_Avoid_: budget (bare), capacity (bare)

**Budget Constraint**:
The hard upper-bound rule `Σ n(v) ≤ B_total` — total anchors never exceed the
Capacity Budget. Enforced in steady state by a three-part invariant: (i) the
reallocation conserves exactly (slots freed = slots filled), (ii) the total stays
`≤ B_total`, and (iii) when capacity is fully demand-justified (binding) the total
equals `B_total`. The slack below `B_total` (plateau) is honest, not padded with
low-value anchors.
_Avoid_: budget (bare), conservation, equality constraint

**Cell Occupancy** (`n(v)`):
The number of anchors a Control Cell currently holds (anchors with Cell Membership
in it) — its present state.
_Avoid_: capacity (bare), count

**Target Capacity** (`c*(v)`):
The number of anchors the controller assigns a Control Cell — the allocation it
grows or prunes toward.
_Avoid_: capacity (bare), target

## Demand

**Anchor Demand** (`s(a)`):
The per-anchor raw demand signal — unitless, non-negative, comparable. The
primitive the Demand Producer emits (`error × visibility`, or a semantic score);
everything else is derived from it.
_Avoid_: importance score, weight

**Demand Score** (`d(v)`):
The per-Control-Cell demand — the sum of Anchor Demand over the cell's members
(`d(v) = Σ_{a∈v} s(a)`). Unitless, non-negative, cross-cell-comparable; relative
and rank-meaningful only, no physical-quantity interpretation. The single (L1)
normalization that turns it into an allocation weight lives in the controller.
_Avoid_: importance value, saliency

**Demand Field**:
The collection of Demand Scores over all Control Cells — produced by reducing
Anchor Demand over Cell Membership (in the Partition layer) — the controller's sole
input describing "where" detail is needed.
_Avoid_: importance map, heatmap (the heatmap is a visualization of it)

**Demand Producer**:
The swappable, **partition-agnostic** component that emits Anchor Demand (per
anchor) — it knows nothing of cells or `control_level`. Current implementation is
error/visibility-driven; a future one is semantic/instance-driven.
_Avoid_: scorer, estimator
