# Issue: Walking skeleton — package layout + ABC interfaces + degenerate closed-loop path

**Status:** ready-for-agent

## What to build

Create the `ocbgs/` package skeleton and wire a **degenerate (no-op) closed-loop controller path** into `adjust_anchor`, so every module boundary, import seam, and integration call site is exercised end-to-end before any real logic lands. The degenerate path passes through all four pipeline units but changes no training behaviour.

**Lazy rasterizer import seam.** The pure-logic modules (`demand/`, `partition/`, `controller/`) must not transitively import the CUDA rasterizer. Establish a lazy-import or optional-import pattern in `ocbgs/` so that `import ocbgs.controller` succeeds on Windows with no CUDA toolkit installed.

**Package layout.** Create `ocbgs/__init__.py`, `ocbgs/demand/__init__.py`, `ocbgs/partition/__init__.py`, `ocbgs/controller/__init__.py` (each with at most a docstring and a re-export of the public ABC).

**ABC interfaces (stub implementations).** Define four abstract base classes (or protocols) matching the ADR contracts, each returning identity/trivial values:

- `DemandProducer` ABC — `produce(scene, stats) -> s(a) Tensor[N]`. Stub returns `torch.ones(N)`.
- `Partition` ABC — `set_control_level(anchor_positions) -> int`, `cell_id(anchor_positions) -> Tensor[N]`, `reduce(anchor_positions, weights, exclude=None) -> (cell_ids, values)`. Stub returns a single global cell.
- `BudgetController` ABC — `plan(cell_ids, d_A, occupancy, B_total, d_B=None) -> ReallocationPlan`. `d_B` is an optional second demand field (Source B, issue 06); `phase` is determined internally by temporal state (issue 03b), not passed by the caller. Stub returns identity (delta=0, phase="ramp").
- `ReallocationPlan` — a dataclass/NamedTuple with fields `cell_ids`, `delta`, `phase`, `c_target`.

**Degenerate `adjust_anchor` controller path.** In `gaussian_model.py`, add a `controller_active(iteration)` gate that enters the degenerate path only for a single test step (e.g. `iteration == opt.update_until - 1`). The degenerate path:

1. Calls `self.demand_producer.produce(...)` → gets `torch.ones`.
2. Calls `self.partition.reduce(...)` → gets one global cell.
3. Calls `self.controller.plan(...)` → gets identity plan (delta=0).
4. Calls native `_native_adjust_anchor` (unchanged grow/prune — no behaviour change).

`demand_producer`, `partition`, `controller` are constructed once and attached to `self` during model init. `train.py` call site is unchanged.

## Acceptance criteria

- [ ] `import ocbgs.controller` succeeds on Windows with no CUDA toolkit (lazy rasterizer import)
- [ ] `import ocbgs.demand`, `import ocbgs.partition`, `import ocbgs.controller` all succeed
- [ ] ABC contract tests pass locally: each stub can be instantiated, each method call returns the documented shape/dtype
- [ ] `ReallocationPlan` type is defined and importable
- [ ] One full training step on Linux server enters the degenerate controller path, runs end-to-end without error, and exits with training behaviour identical to native Octree-GS
- [ ] The degenerate path is off by default (gated); native path is byte-equivalent to original Octree-GS

## Blocked by

None — can start immediately.
