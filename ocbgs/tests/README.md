# Walking Skeleton Tests — Run Instructions

This suite (`test_00_walking_skeleton.py`) validates the 00 walking-skeleton acceptance criteria: lazy CUDA loading, the demand/partition/controller ABC contracts and their stubs, the degenerate controller closed-loop gating, `B_total` plumbing, and fixed-seed reproducibility.

## Test environment split

The suite is designed to run in two places with the same command:

- **Local (CPU / torch-cpu)** — pure-logic tests run; every test that needs the compiled CUDA extensions imports `scene.gaussian_model` / `gaussian_renderer` inside a `try/except ImportError` and is reported as **SKIPPED**. A clean local run is therefore "passed + skipped, zero failures".
- **Server (Linux + CUDA)** — the CUDA extensions are present, nothing skips, and the full suite runs.

The degenerate closed-loop *end-to-end* step (acceptance #5) and a real `bash setup.sh` build are only exercised on the server; the unit tests assert structure and wiring, not a full training step.

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

Run from the repository root. `setup.sh` creates the `ocbgs` conda env from `environment.yml`, initializes the GLM submodule (`git submodule update --init`), then `pip install -e .` builds `diff-gaussian-rasterization` and `simple-knn` from the vendored sources.

```bash
bash setup.sh                 # or: bash setup.sh <env-name>
conda activate ocbgs
```

A successful build ends with the `=== Setup complete ===` banner and no compiler errors.

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

## What the suite covers

- **Package imports without CUDA** — `demand`, `partition`, `controller` import with only `torch` + `abc`; `gaussian_renderer` does no top-level `diff_gaussian_rasterization` import (`_lazy_rasterizer`).
- **ABC enforcement** — the three abstract base classes cannot be instantiated.
- **Stubs** — `StubDemandProducer`, `StubPartition`, `StubBudgetController` produce the contracted shapes/dtypes; `ReallocationPlan` is a dataclass with `cell_ids` / `delta` / `phase` / `c_target`.
- **Controller gating** — `controller_active` fires exactly once and only inside `[update_from, update_until]`.
- **Occupancy wiring** — per-anchor `cell_id` membership is counted and re-aligned to the unique `cell_ids` order.
- **`B_total` plumbing** — present on `OptimizationParams` (default `-1`) and on the `GaussianModel` constructor.
- **Seed support** — `safe_state(silent, seed=0)` and the `--seed` argument.
- **Setup files** — `environment.yml` and `setup.sh` exist and carry no duplicate CUDA-submodule references.

## Notes

- If a CUDA test unexpectedly skips on the server, the extensions did not build — re-check the `setup.sh` output before trusting a green run.
- `setup.sh` must be run from the repository root (it `cd`s into `ocbgs/submodules/...`).
