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

The pytest suite asserts wiring, not a training step, so it cannot by itself prove the two runtime-behaviour criteria: #5 (a full training step enters the degenerate controller path, runs end-to-end without error, behaviour identical to native Octree-GS) and the byte-equivalence clause of #6 (native path byte-equivalent to original Octree-GS). Those are verified by the two complementary checks below. Note that the CUDA rasterizer backward uses `atomicAdd`, so two separate training runs are **not** bit-reproducible even with a fixed seed — never prove equivalence by diffing the saved `.ply` of two independent runs. Both checks below sidestep that non-determinism.

### Part A — Static source equivalence (#6)

In the training-mutation path, the only difference between OcbGS and original Octree-GS is the gated block at the top of `adjust_anchor`, and that block computes a `ReallocationPlan` which is then discarded (never applied); the stubs it calls touch no RNG and mutate no model state. So the native trajectory is provably unaffected. Reproduce the diff on the server (or locally):

```bash
cd ~/OcbGS
diff <(sed -n '/def anchor_growing/,/def prune_anchor/p' refered_repo/Octree-GS/scene/gaussian_model.py) \
     <(sed -n '/def anchor_growing/,/def prune_anchor/p' ocbgs/scene/gaussian_model.py)
```

**Pass:** the only net change is the inserted `if self.controller_active(iteration): ... plan = self.controller.plan(...)` block (~10 lines); every other line of `anchor_growing` / prune / save / load is identical (the rest of the diff is pure line-number shift).

### Part B — Runtime no-op assertion (#5)

This runs one genuine end-to-end training step on the server through the real rasterizer + optimizer, and proves the gated block is a no-op **within a single run** (deterministic, no cross-run atomics).

**1. The check is built in, off by default.** `adjust_anchor` in `ocbgs/scene/gaussian_model.py` contains an opt-in guard: when the env var `OCBGS_VERIFY_DEGENERATE=1` is set it snapshots all model tensors on entry to the gated block and asserts, after the discarded `plan` is computed, that none changed. Unset (the default) the guard is skipped entirely, so the gated block stays a byte-level no-op — nothing to edit or revert.

**2. Fetch a minimal COLMAP scene** from Hugging Face onto the fast data disk (`/root/autodl-tmp`, kept off the small system disk). MipNeRF360 `garden` is COLMAP-format (robust — the Blender reader in `dataset_readers.py` crashes on a scene with no `.ply`); only its `sparse/` model and the 1/8-downsampled `images_8` are needed (< 100 MB):

The `hf` CLI replaces the deprecated `huggingface-cli`; pass one `--include` per pattern (a second bare pattern is parsed as a positional filename, not an include). On AutoDL, plain huggingface.co is slow — use one of the two acceleration options below.

```bash
pip install -U "huggingface_hub[cli]"
```

Option 1 — accelerate huggingface.co for the whole session (`source`, not execute; per-session, re-run in each new terminal):

```bash
source /etc/network_turbo
hf download mileleap/mipnerf360 --repo-type dataset --include "garden/sparse/**" --include "garden/images_8/**" --local-dir /root/autodl-tmp/m360
```

Option 2 — use the hf-mirror.com mirror, no global proxy (set `HF_ENDPOINT` for just this command):

```bash
HF_ENDPOINT=https://hf-mirror.com hf download mileleap/mipnerf360 --repo-type dataset --include "garden/sparse/**" --include "garden/images_8/**" --local-dir /root/autodl-tmp/m360
```

**3. Run a short training** with the guard enabled, firing the gate early (default `update_from` is 1500; override so it fires at iteration 20):

```bash
cd ~/OcbGS/ocbgs
OCBGS_VERIFY_DEGENERATE=1 python train.py -s /root/autodl-tmp/m360/garden --ds 8 -m /root/autodl-tmp/verify_run \
  --iterations 60 --start_stat 5 --update_from 10 --update_interval 10 \
  --update_until 50 --test_iterations 60 --save_iterations 60 --seed 0
```

Use `--ds 8`, **not** `-i images_8`: the Octree-GS COLMAP reader (`dataset_readers.py`) ignores `--images` and selects the image folder from `--ds` (`ds=1 → images/`, `ds=8 → images_8/`). The gate fires when `iteration > update_from (10)` and `iteration % update_interval (10) == 0`, i.e. iteration 20. (To use a different resolution, fetch the matching `images_N` folder and pass `--ds N`.)

**Pass (all three):**
1. stderr shows `[VERIFY] degenerate path ENTERED at iter 20` — the path was actually entered.
2. immediately followed by `[VERIFY] degenerate path is a byte-level NO-OP` — the gated block mutated no model tensor.
3. training (and the subsequent render + metric evaluation) reaches the end with no exception and writes `point_cloud/` + mlp checkpoints under `/root/autodl-tmp/verify_run`.

**4. Clean up** (nothing to revert — the guard is committed and off by default): `rm -rf /root/autodl-tmp/m360 /root/autodl-tmp/verify_run`.

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
