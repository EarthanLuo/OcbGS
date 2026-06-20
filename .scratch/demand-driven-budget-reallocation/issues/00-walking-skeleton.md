# Issue: Walking skeleton — package layout + ABC interfaces + degenerate closed-loop path

**Status:** done

## What to build

Create the `ocbgs/` package skeleton and wire a **degenerate (no-op) closed-loop controller path** into `adjust_anchor`, so every module boundary, import seam, and integration call site is exercised end-to-end before any real logic lands. The degenerate path passes through all four pipeline units but changes no training behaviour.

**Lazy rasterizer import seam.** The pure-logic modules (`demand/`, `partition/`, `controller/`) must not transitively import the CUDA rasterizer. Establish a lazy-import or optional-import pattern in `ocbgs/` so that `import ocbgs.controller` succeeds on Windows with no CUDA toolkit installed.

**Package layout.** Create `ocbgs/__init__.py`, `ocbgs/demand/__init__.py`, `ocbgs/partition/__init__.py`, `ocbgs/controller/__init__.py` (each with at most a docstring and a re-export of the public ABC).

**ABC interfaces (stub implementations).** Define four abstract base classes (or protocols) matching the ADR contracts, each returning identity/trivial values:

- `DemandProducer` ABC — `produce(scene, stats) -> s(a) Tensor[N]`. Stub returns `torch.ones(N)`.
- `Partition` ABC — `set_control_level(anchor_positions) -> int`, `cell_id(anchor_positions) -> Tensor[N]`, `reduce(anchor_positions, weights, exclude=None) -> (cell_ids, values)`. Stub returns a single global Control Cell.
- `BudgetController` ABC — `plan(cell_ids, d_A, occupancy, B_total, d_B=None) -> ReallocationPlan`. `d_B` is an optional second demand field (Source B, issue 06); `phase` is determined internally by temporal state (issue 03b), not passed by the caller. Stub returns identity (delta=0, phase="ramp").
- `ReallocationPlan` — a dataclass/NamedTuple with fields `cell_ids`, `delta`, `phase`, `c_target`.

**Degenerate `adjust_anchor` controller path.** In `gaussian_model.py`, add a `controller_active(iteration)` gate that enters the degenerate path only for a single test step (e.g. `iteration == opt.update_until - 1`). The degenerate path:

1. Calls `self.demand_producer.produce(...)` → gets `torch.ones`.
2. Calls `self.partition.reduce(...)` → gets one global Control Cell.
3. Calls `self.controller.plan(...)` → gets identity plan (delta=0).
4. Calls native `_native_adjust_anchor` (unchanged grow/prune — no behaviour change).

`demand_producer`, `partition`, `controller` are constructed once and attached to `self` during model init. `train.py` call site is unchanged.

## Acceptance criteria

- [x] `import ocbgs.controller` succeeds on Windows with no CUDA toolkit (lazy rasterizer import)
- [x] `import ocbgs.demand`, `import ocbgs.partition`, `import ocbgs.controller` all succeed
- [x] ABC contract tests pass locally: each stub can be instantiated, each method call returns the documented shape/dtype
- [x] `ReallocationPlan` type is defined and importable
- [x] One full training step on Linux server enters the degenerate controller path, runs end-to-end without error, and exits with training behaviour identical to native Octree-GS
- [x] The degenerate path is off by default (gated); native path is byte-equivalent to original Octree-GS
- [x] `environment.yml` (loose pins, tolerant of arbitrary PyTorch version) and `setup.sh` (create env, build Octree-GS CUDA submodule) are created at project root (spec §7.3)
- [x] Fixed random seed support for baseline runs; Octree-GS `arguments/` config system records every experiment setting (spec §7.3)

## Blocked by

None — can start immediately.

## Comments

### 2026-06-20 — Server test run (6/8 acceptance criteria verified)

Full `test_00_walking_skeleton.py` suite run on the Linux + CUDA server (autodl image: PyTorch 2.5.1 + CUDA 12.4, Python 3.12.3, RTX 4090 / sm_89): **34 passed, 0 skipped**. Zero skips confirms the CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`) compiled and the `scene.gaussian_model` / `gaussian_renderer` import seams resolve.

Setup retargeted to reuse the image's preinstalled torch (commit `4838fb7`): `setup.sh` no longer runs `conda env create`; it installs `torch-scatter` from the matching pyg wheel and builds the CUDA extensions with `TORCH_CUDA_ARCH_LIST=8.9`. Root `environment.yml` demoted to a reference manifest. The stale `ocbgs/environment.yml` (py3.7 / torch1.12 Octree-GS leftover) and the vestigial `ocbgs/.git` submodule pointer were removed.

Verified by this run: criteria #1–#4 (imports, lazy rasterizer seam, ABC contracts, `ReallocationPlan`), #7 (setup files present + build succeeds), #8 (seed support / `--seed`). The gating half of #6 is covered by `TestControllerActive` (fires exactly once, only inside `[update_from, update_until]`).

**Still pending — not covered by the pytest suite (by design it asserts wiring, not a training step):**
- #5: one full training step on the server entering the degenerate controller path end-to-end, with behaviour identical to native Octree-GS.
- #6 (second clause): native path byte-equivalence to original Octree-GS — requires the same real training-step comparison as #5.

Status set to `ready-for-human`: the only remaining gate is a human-run training-step / byte-equivalence comparison against native Octree-GS.

### 2026-06-20 — #5 / #6 verified end-to-end (all 8 criteria met, status → done)

The two remaining criteria are now verified on the server (MipNeRF360 `garden`, `--ds 8`, 60 iterations, gate forced early via `--update_from 10 --update_interval 10`, fixed `--seed 0`):

- **#6 static equivalence (Part A):** diff of `adjust_anchor` / `anchor_growing` / prune / save / load against `refered_repo/Octree-GS` shows the only net change is the gated block, whose `ReallocationPlan` is computed then discarded; the stubs it calls touch no RNG and mutate no model state.
- **#5 + #6 runtime no-op (Part B):** with `OCBGS_VERIFY_DEGENERATE=1` the degenerate path printed `[VERIFY] degenerate path ENTERED at iter 20` followed by `[VERIFY] degenerate path is a byte-level NO-OP` — entry observed, and every model tensor byte-identical across the gated block. The full pipeline (train → render → eval) completed without error (PSNR 18.75 / SSIM 0.587 on the 60-iter smoke run).

The verification check is committed and off by default (env-guarded in `adjust_anchor`, commit `a8a8874`), so the gated block remains a byte-level no-op in normal runs; the procedure is documented in `tests/README.md` (commit `caa8239`). A NumPy >= 1.24 incompatibility found during this run (`np.int` in `load_ply_sparse_gaussian`, which crashed render/eval) was fixed in the same commit. Procedure and findings are reproducible per `tests/README.md` "Degenerate-path equivalence check".
