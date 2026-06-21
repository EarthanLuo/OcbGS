# Walking Skeleton Tests — Run Instructions

This suite (`test_00_walking_skeleton.py`) validates the 00 walking-skeleton acceptance criteria: lazy CUDA loading, the demand/partition/controller ABC contracts and their stubs, the degenerate controller closed-loop gating, `B_total` plumbing, and fixed-seed reproducibility.

## Test environment split

The suite is designed to run in two places with the same command:

- **Local (CPU / torch-cpu)** — pure-logic tests run; every test that needs the compiled CUDA extensions imports `scene.gaussian_model` / `gaussian_renderer` inside a `try/except ImportError` and is reported as **SKIPPED**. A clean local run is therefore "passed + skipped, zero failures".
- **Server (Linux + CUDA)** — the CUDA extensions are present, nothing skips, and the full suite runs.

The degenerate closed-loop *end-to-end* step (acceptance #5) and a real `bash setup.sh` build are only exercised on the server; the unit tests assert structure and wiring, not a full training step. The end-to-end runtime checks for #5 / #6 are documented separately under [Degenerate-path equivalence check](#degenerate-path-equivalence-check-acceptance-5--6).

## Workflow: local push → server pull → run

### 1. Local — commit and push

Build artifacts (`*.so`, `*.egg-info/`, `build/`, `__pycache__/`) and the SIBR viewer are excluded by `.gitignore`; the server recompiles the CUDA extensions from source, so nothing binary is pushed. The vendored CUDA sources under `submodules/diff-gaussian-rasterization` **are** committed. GLM — the rasterizer's header-only dependency — is a pinned **git submodule** at `ocbgs/submodules/diff-gaussian-rasterization/third_party/glm` (g-truc/glm @ 1.0.1), so the server must initialize it before building. `setup.sh` does this automatically; alternatively clone with `--recursive` (see step 2).

```bash
git add -A
git status            # sanity-check: no SIBR_viewers/, no *.so, no *.egg-info/
git commit -m "<message>"
git push origin feat/00-walking-skeleton
```

### 2. Server — pull

```bash
# first time (no --recursive — refered_repo/ submodules are for development only; setup.sh initializes GLM separately)
git clone https://github.com/EarthanLuo/OcbGS.git OcbGS
cd OcbGS
git checkout feat/00-walking-skeleton

# subsequent updates
cd OcbGS
git fetch origin
git checkout feat/00-walking-skeleton
git pull
git submodule update --init ocbgs/submodules/diff-gaussian-rasterization/third_party/glm
```

### 3. Server — build the environment (compiles the CUDA extensions)

`setup.sh` **reuses the image's pre-installed PyTorch** (PyTorch 2.5.1 + CUDA 12.4, Python 3.12, RTX 4090 / sm_89) — it does *not* create a conda env and does *not* reinstall torch. Activate that torch env first, then run the script from the repository root. It installs `torch-scatter` from the matching pyg wheel, the pip dependencies, initializes the GLM submodule (`git submodule update --init`), and builds `diff-gaussian-rasterization` and `simple-knn` from the vendored sources with `TORCH_CUDA_ARCH_LIST=8.9`.

```bash
conda activate <image-torch-env>   # the env that already has torch 2.5.1+cu124
bash setup.sh                      # run from the repository root
```

A successful build ends with the `=== Setup complete ===` banner and no compiler errors. To build for a different GPU, override the arch: `TORCH_CUDA_ARCH_LIST="8.6" bash setup.sh`.

### 4. Server — run the full suite

```bash
cd ocbgs
python -m pytest tests/test_00_walking_skeleton.py -v
```

Expected on a CUDA box: **all tests pass, none skipped**.

### 5. (Optional) Local sanity check — CPU only

The same command runs locally without CUDA; use it before pushing to catch pure-logic regressions early.

```bash
cd ocbgs
python -m pytest tests/test_00_walking_skeleton.py -v
```

Expected locally: logic tests pass; the CUDA-dependent tests (those importing `scene.gaussian_model` / `gaussian_renderer`) report **SKIPPED**. Any **FAILED** is a real problem.

## Degenerate-path equivalence check (acceptance #5 / #6)

**[Superseded by issue 05.]** The degenerate no-op controller path and its `OCBGS_VERIFY_DEGENERATE` snapshot guard described below were removed in issue 05. The controller path now fully applies the `ReallocationPlan` (prune, grow, accumulator updates). The gated block is no longer discarded.

### Part A — Static source equivalence (#6) → verified via source diff (issue 05)

After issue 05, native byte-equivalence is verified statically: `_native_adjust_anchor` (approx. gaussian_model.py) is a verbatim copy of the original native `adjust_anchor` body from `refered_repo/Octree-GS/scene/gaussian_model.py` lines 852–904, and `anchor_growing` (approx. lines 1063–1196) is the unmodified upstream implementation. Confirm with:

```bash
diff <(sed -n '/def _native_adjust_anchor/,/def adjust_anchor/p' ocbgs/scene/gaussian_model.py | sed '1d;$d') \
     <(sed -n '/def adjust_anchor/,/def save_mlp_checkpoints/p' refered_repo/Octree-GS/scene/gaussian_model.py | sed '1d;$d')
```

The diff shows only line-number shifts and added controller methods; the native path code is bit-for-bit identical.

### Part B — Runtime no-op assertion (#5)

**[Superseded by issue 05 — REMOVED.]** The `OCBGS_VERIFY_DEGENERATE` env-var guard and the degenerate-path snapshot block no longer exist in `gaussian_model.py`. The controller path now executes real mutations. The scene fetch and short-training commands below are retained for reference but the env var is a no-op.

## What the suite covers

- **Package imports without CUDA** — `demand`, `partition`, `controller` import with only `torch` + `abc`; `gaussian_renderer` does no top-level `diff_gaussian_rasterization` import (`_lazy_rasterizer`).
- **ABC enforcement** — the three abstract base classes cannot be instantiated.
- **Stubs** — `StubDemandProducer`, `StubPartition`, `StubBudgetController` produce the contracted shapes/dtypes; `ReallocationPlan` is a dataclass with `cell_ids` / `delta` / `phase` / `c_target`.
- **Controller gating** — `controller_active` activates every step during the reallocation window `(post-unlock, update_until]` [Superseded by issue 05].
- **Occupancy wiring** — per-anchor `cell_id` membership is counted and re-aligned to the unique `cell_ids` order.
- **`B_total` plumbing** — present on `OptimizationParams` (default `-1`) and on the `GaussianModel` constructor.
- **Seed support** — `safe_state(silent, seed=0)` and the `--seed` argument.
- **Setup files** — `environment.yml` and `setup.sh` exist and carry no duplicate CUDA-submodule references.

## Notes

- If a CUDA test unexpectedly skips on the server, the extensions did not build — re-check the `setup.sh` output before trusting a green run.
- `setup.sh` must be run from the repository root (it `cd`s into `ocbgs/submodules/...`).

## issue 03b — TemporalBudgetController

All 24 tests in `test_03b_controller_temporal_layer.py` are pure-logic (no CUDA, no `gaussian_model` import) and run locally. To run:

```bash
cd ocbgs
python -m pytest tests/test_03b_controller_temporal_layer.py -v
```

**Pass:** 24 passed, 0 skipped, 0 failed.

## issue 04 — Actuator pure helpers

The two helpers (`_opacity_dead_mask`, `_lowest_sa_in_surplus`) are `@staticmethod` pure PyTorch functions in `gaussian_model.py`, but the module's import chain (`torch_scatter`, `simple_knn._C`, `plyfile`, `einops`, `scene.embedding` → `PIL`) requires the full CUDA environment and packages only present after `bash setup.sh`. The 16 tests in `test_04_actuator_pure_helpers.py` must be run on the server.

### Server verify

```bash
cd ~/OcbGS/ocbgs
python -m pytest tests/test_04_actuator_pure_helpers.py -v
```

**Pass:** 16 passed, 0 skipped, 0 failed. (On Windows / CPU, all 16 SKIP with `pytest.skip("GaussianModel import requires CUDA environment")`.)

### Local verify

The helper function bodies contain no CUDA and can be validated by inspection:

- `_opacity_dead_mask`: `mean_opacity = opacity_accum / (anchor_demon + 1e-8)` → `(mean_opacity < min_opacity) & (anchor_demon > maturity_min)` — reproduces the existing `gaussian_model.py:934-936` GC semantics exactly.
- `_lowest_sa_in_surplus`: iterates surplus cells (`delta < 0`), uses `torch.topk(s_in_cell, min(count, n_in_cell), largest=False)` to select the lowest-`s(a)` anchors.

### Issue 05 integration notes

When wiring issue 05 (`adjust_anchor` rewrite):

1. `_opacity_dead_mask` caller must pass `maturity_min = check_interval * success_threshold` (the helper owns no policy defaults — single source of truth at the `adjust_anchor` signature).
2. The native `gaussian_model.py:950-952` accumulator reset (`opacity_accum`, `anchor_demon` zeroed for mature anchors) is NOT part of this helper — that is issue 05's integration responsibility.
