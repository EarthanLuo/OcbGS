# Issue: `set_control_level` emits a misleading error when B_total is too small for the scene

**Status:** DONE

## Context

Surfaced while verifying issue 05 on the server. Running the controller path with `--B_total 500` on garden (ds=8, ~492K initial anchors) crashes with:

```
ValueError: Σfloor (74194) > B_total (500); control_level derivation should preclude this
```

raised from `controller._allocate` (`ocbgs/controller/__init__.py`). The root cause is **not** a controller bug: even the coarsest octree level has 74194 occupied Control Cells, so with `floor=1` the minimum representable anchor count (74194) already exceeds `B_total=500`. The budget is simply infeasible for the scene.

The behaviour is correct (it refuses an impossible budget), but two things are wrong about how it surfaces:

1. **Misleading message.** `OctreePartition.set_control_level` (`ocbgs/partition/__init__.py`) silently returns the coarsest level in its fallback branch, then the controller raises a message blaming "control_level derivation" — pointing the reader at a derivation bug rather than the real cause (B_total too small). The `set_control_level` fallback does not itself check `floor * N_active ≤ B_total`.
2. **Late failure.** The infeasibility is only detected one call later inside the controller, after partitioning, rather than at derivation time.

This is pre-existing partition behaviour (issues 02/03); issue 05 is just the first caller to exercise the controller path on a real scene and trip it.

## What to build

In `OctreePartition.set_control_level`, after the feasibility loop and the A_min fallback, detect the infeasible-budget case and raise a clear, actionable error at derivation time:

- When the chosen (coarsest available) level still has `floor * N_active > B_total`, raise a `ValueError` naming the real cause, e.g.: `"B_total={B_total} too small for scene: coarsest control level has {N_active} occupied cells; need B_total >= floor*{N_active}={floor*N_active}."`
- Do **not** change the fallback level-selection direction — it already correctly takes the coarsest level meeting `A_min` (level 0 is the coarsest; `range(levels)` iterates coarse→fine and `break` takes the first match). The earlier suspicion of a "finest-first" bug was a level-convention misreading; reversing iteration would pick the finest level and make it strictly worse.

## Acceptance criteria

- [x] `set_control_level` raises a `ValueError` whose message names `B_total`, `N_active`, and the `floor*N_active` lower bound when the budget cannot give every occupied coarsest-level cell its floor.
- [x] The error is raised from `set_control_level` (derivation time), not deferred to `controller._allocate`.
- [x] A feasible-budget call (e.g. `B_total >= floor*N_active`) is unaffected — derivation returns the same level as before.
- [x] Fallback level selection is unchanged (still coarsest level meeting `A_min`); a unit test asserts it picks the coarsest, not the finest.
- [x] Pure-logic unit tests only (no CUDA); runnable locally against `OctreePartition` directly.

## Notes

- Keep `floor`, `rho_min`, `A_min` as the existing derivation knobs; this issue only adds an explicit feasibility guard + clearer message, no policy change.
