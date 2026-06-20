# Issue: Actuator pure helpers — `_opacity_dead_mask` + `_lowest_sa_in_surplus`

**Status:** ready-for-agent  <!-- helpers implemented in gaussian_model.py, 16 tests written in test_04_actuator_pure_helpers.py; all tests require CUDA server to execute (see note under Acceptance criteria) -->

## What to build

Implement two pure-PyTorch helper functions in `gaussian_model.py` that will be called by the integrated `adjust_anchor` (issue 05). Neither function mutates optimizer state; both are unit-testable with synthetic anchor tensors.

**These are hooks in `gaussian_model.py`, NOT a separate `ocbgs/actuator/` module** (ADR-0001 D7).

**`_opacity_dead_mask(opacity_accum: Tensor, min_opacity: float) -> Tensor[bool]`**

Identifies anchors whose opacity has collapsed below a threshold, marking them as "dead" for garbage collection (ADR-0004 Step 0).

- Inputs: `opacity_accum: Tensor[N_anchors]` (per-anchor accumulated opacity from `training_statis`), `min_opacity: float` (threshold, from the existing Octree-GS parameter)
- Output: boolean mask `[N_anchors]`, True = dead (should be pruned)
- Threshold: `opacity_accum < min_opacity`
- Pure function — receives all inputs explicitly, unit-testable with synthetic tensors (no `self.` access)
- Accounting rationale (from ADR-0004): GC is orthogonal to demand and must stay accounted. A dead anchor in a deficit Control Cell is never removed by demand-prune (that Control Cell is growing, not pruning), so it permanently wastes a slot without GC.

**`_lowest_sa_in_surplus(plan: ReallocationPlan, s_a: Tensor, anchor_cell_ids: Tensor) -> Tensor[bool]`**

In each surplus Control Cell (`δ(v) < 0`), selects the `|δ(v)|` anchors with the lowest Anchor Demand `s(a)` and returns their membership in the prune set.

- Inputs: `ReallocationPlan` (from BudgetController), `s_a: Tensor[N_anchors]` (from DemandProducer), `anchor_cell_ids: Tensor[N_anchors]` (per-anchor cell ids from `Partition.cell_id()` — the anchor→Control Cell mapping required to bin anchors into surplus Control Cells)
- Output: boolean mask `[N_anchors]`, True = should be demand-pruned
- Per surplus Control Cell: sort anchors by `s(a)` ascending, take the `|δ(v)|` lowest
- Anchors NOT in surplus Control Cells are always False
- `s(a)` is the second fan-out consumer of `s(a)` in the architecture (ADR-0001 constraint 1): the first is Partition's reduction `s(a) → d(v)` (issue 02); the Actuator uses `s(a)` for prune ranking only

**Tripwire — two ranking signals, never confused (ADR-0005 §4):**
- Grow ranks candidates by **proposing offset gradient** (handled by `anchor_growing_capped` in issue 05)
- Prune ranks established anchors by **`s(a)`** (handled here)
- `s(a)` is undefined for not-yet-created grow candidates — it is a prune-side signal only

## Acceptance criteria

- [ ] `_opacity_dead_mask`: all anchors with `opacity_accum < min_opacity` → True; all above → False
- [ ] `_opacity_dead_mask`: zero anchors → returns empty (all-False) mask, no crash
- [ ] `_opacity_dead_mask`: all-dead → returns all-True mask
- [ ] `_lowest_sa_in_surplus`: single surplus Control Cell with `|δ| = 3` and 5 anchors → selects the 3 anchors with lowest `s(a)` (given correct `anchor_cell_ids` assigning all 5 to that Control Cell)
- [ ] `_lowest_sa_in_surplus`: multiple surplus Control Cells — each Control Cell independently selects its `|δ(v)|` lowest-`s(a)` anchors (verified with distinct `anchor_cell_ids`) 
- [ ] `_lowest_sa_in_surplus`: deficit Control Cell (δ > 0) → no anchors selected (always False)
- [ ] `_lowest_sa_in_surplus`: surplus Control Cell with `|δ| > n(v)` (plan asks to prune more than exist) → selects all anchors in that Control Cell (graceful, no crash)
- [ ] `_lowest_sa_in_surplus`: `|δ| = 0` → no anchors selected, empty mask
- [ ] Both functions are pure PyTorch (no optimizer access, no CUDA); unit-testable with synthetic tensors
- [ ] Neither function mutates any anchor state or optimizer state

**Note on local testability:** The two helper functions themselves are pure `@staticmethod` PyTorch (no CUDA, no optimizer access). However, they live in `gaussian_model.py` which has module-level imports of `torch_scatter`, `simple_knn._C`, `plyfile`, `einops`, and `scene.embedding` → `scene.dataset_readers` → `PIL` — a deep import chain that requires CUDA and several packages not available on Windows. As a result, the 16 unit tests are skipped on Windows (`pytest.skip` on `ImportError`) and must be verified on the CUDA server. See `ocbgs/tests/README.md` for server test procedure.

## Blocked by

- 02-partition-cell-membership (needs `cell_id` to bin anchors into Control Cells for per-Control-Cell prune ranking)
- 03a-controller-static-allocator (needs `ReallocationPlan` type with `cell_ids` and `delta` fields)
