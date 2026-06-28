## Agent skills

### Issue tracker

Issues are local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Language

Interactive communication with the user is in Simplified Chinese. All content written to markdown files (CONTEXT.md, ADRs, docs, reports) is in English.

### Markdown formatting

**Do NOT hard-wrap paragraphs** in markdown files (never break lines at a fixed column like 80 chars). Let the renderer wrap text to the viewer's window width. A paragraph is a single long line; paragraphs are separated by blank lines.

When unwrapping a hard-wrapped file, preserve structural elements as-is:
- Headings (`#`), code fences (` ``` `), tables (`|`), metadata lines (`**Field:** …`)
- List items: merge each item's continuation lines but keep each item on its own line

For mixed blocks (intro sentence followed by numbered list in the same paragraph block), split the intro as one merged line and keep the numbered items on separate lines so the list renders correctly.

### Reference implementations

When implementing any algorithm or module, consult the corresponding reference implementations first to avoid reinventing the wheel. The four key reference repositories are:

- `refered_repo/gsplat/` — Official 3DGS CUDA rasterizer and training framework (Kerbl et al., 2023)
- `refered_repo/Octree-GS/` — Octree-based LOD-structured 3D Gaussians (Ren et al., 2024)
- `refered_repo/CLoD-GS/` — Continuous Level-of-Detail via 3D Gaussian Splatting (Li et al., 2026)
- `refered_repo/FastGS/` — Training 3DGS in 100 Seconds via Multi-View Consistency (Paliwal et al., 2026)

Similarly, for theoretical foundation and algorithmic details, reference the papers in `docs/literature/text/` and `docs/literature/papers/` before designing from scratch. `docs/literature/text/` contains plain-text extractions from the PDF papers stored in `docs/literature/papers/`.

### Knowledge graph

`.understand-anything/knowledge-graph.json` contains a structured knowledge map of all four reference repositories — files, classes, functions, and their interrelationships. Consult it before navigating the reference code to quickly locate relevant implementations.

### Development workflow

Write code test-first using TDD (red-green-refactor): write a failing test case before the implementation, then write the minimal code to make it pass.

Run only tests that carry no risk of environment conflict — e.g. pure-logic unit tests with no GPU/CUDA, server, or heavy-dependency requirements. For any test that might conflict with the environment (GPU/CUDA, server-only resources, large datasets, long-running training, etc.), do NOT run it: hand it to the user with exact run instructions (the command to invoke and what a passing result looks like) and let the user run it on the server.

After finishing a piece of code, self-review it once first — check for correctness, edge cases, and adherence to project conventions — and only then hand it to the user for review.

### Experiment workflow (knob tuning — fast inner loop, slow outer loop)

Do NOT tune knobs (e.g. `--grow_relax_scale`, `k_cap`, `rate_limit`) with full-length training runs. Split the questions: *"did the mechanism move?"* — budget fill ratio (`final anchors / B_total`), allocation change, no-crash/no-OOM — is answerable with a **short smoke** (~3k iters + compressed controller window, e.g. `--iterations 3000 --update_from 500 --update_interval 100 --update_until 3000`) in minutes; the fill ratio is roughly knob-determined and horizon-independent, so the smoke extrapolates to the full run. Only *"is quality better?"* (PSNR/SSIM/LPIPS) needs a full run, and each arm is run **once**, after the knob is locked. Rule: every full run must answer something a smoke cannot (quality); anything about moved/filled/allocation goes to a smoke.

### Single-seed experiments

Every `exp*.sh` script accepts `SEEDS` as an env-var override (all lines use `${SEEDS:-default list}`). Smoke runs and quick mechanism checks use one seed; full quality comparisons use the default multi-seed list. Examples:

```bash
SEEDS="0" bash scripts/exp4_garden.sh                   # single-seed smoke
SEEDS="0 1" bash scripts/exp4_garden.sh                 # two seeds
bash scripts/exp4_garden.sh                              # default: all seeds
```

After an experiment run is no longer needed for rendering, clean up per-seed directories: keep only `results.json` and `outputs.log`, delete everything else (checkpoints, tensorboard events, point cloud snapshots). This reduces a typical 35G experiment tree to ~10MB while preserving all metrics needed for tables and plots. Do NOT delete source datasets under `/root/autodl-tmp/`.

### Acceptance criteria

Each issue's acceptance criteria must be checked item by item before the issue can be marked as passed. When closing an issue, set its **Status:** to DONE.

**Never check off an acceptance criterion whose test was NOT actually executed.** A `pytest.skip` (e.g. due to missing CUDA), an `ImportError` guard, or a manual "this looks correct" all count as NOT executed. Only a test that ran to completion and passed counts. If a test cannot be run in the current environment, leave the checkbox unchecked, set **Status:** back to `ready-for-agent`, and hand the issue to the user with exact run instructions for the server.

### Server & training environment

Pitfalls hit while deploying and verifying on the AutoDL server (image: PyTorch 2.5.1 + CUDA 12.4, Python 3.12, RTX 4090 / sm_89). The full deploy + verification procedure lives in `ocbgs/tests/README.md`.

- **Reuse the image's torch; do not `conda env create`.** The pinned `environment.yml` stack (py3.7 / torch1.12 / cu11.6) is the upstream Octree-GS stack and conflicts with the image. `bash setup.sh` reuses the active torch, installs `torch-scatter` from the matching pyg **pip** wheel (the pyg conda channel has no py3.12 build), and compiles the CUDA extensions with `TORCH_CUDA_ARCH_LIST=8.9`. Root `environment.yml` is a reference manifest, not a create recipe.
- **NumPy ≥ 1.24 removed `np.int` / `np.float` / `np.bool`.** The vendored Octree-GS code still uses them (e.g. `np.int` in `load_ply_sparse_gaussian`), which crashes a render/eval/load path with `AttributeError: module 'numpy' has no attribute 'int'`. Replace with `np.int64` / `np.float32` / `bool`.
- **The COLMAP reader picks the image folder from `--ds`, not `-i/--images`.** `scene/dataset_readers.py` maps `ds=1 → images/`, `ds=8 → images_8/` and ignores `--images`. Pass `--ds N`, not `-i images_N` (the latter silently looks in `images/` and raises `FileNotFoundError`).
- **The Blender reader crashes on a scene with no `.ply`.** `readNerfSyntheticInfo` does `glob("*.ply")[0]` before the existence check, so a fresh nerf_synthetic scene raises `IndexError`. Prefer COLMAP datasets, or pre-place a points ply.
- **The CUDA rasterizer backward uses `atomicAdd` → training is not bit-reproducible run-to-run, even with a fixed seed.** Never prove equivalence by diffing two independent runs' outputs; assert within a single run (snapshot/compare) or statically (source diff).
- **Hugging Face CLI:** `huggingface-cli` is deprecated — use `hf`. `hf download --include` takes one pattern per flag (repeat `--include`; a second bare pattern is parsed as a positional filename). Keep the command on one line — pasted `\` continuations get mangled.
- **AutoDL specifics:** put large data on the fast data disk `/root/autodl-tmp` (system disk `/` is ~30 GB); the box reaches only github / huggingface — `source /etc/network_turbo` or prefix `HF_ENDPOINT=https://hf-mirror.com` to accelerate.
- **Exercising the degenerate controller path:** it is gated to fire once at `update_from` (default 1500); for a short smoke run override `--update_from 10 --update_interval 10 --update_until 50`. Set `OCBGS_VERIFY_DEGENERATE=1` to assert the gated block is a byte-level no-op.
- **`set -e` + `(( var++ ))` silently aborts the experiment scripts.** Bash arithmetic `(( expr ))` returns exit code 1 whenever `expr` evaluates to 0, and post-increment `(( _running++ ))` evaluates to the *old* value — so the first `0 → 1` increment returns 1 and `set -e` kills the script. In the `exp*_*.sh` concurrency loops this happens right after the first background `python train.py &` is launched: the detached job runs to completion (writes `results.json`, prints `Evaluating complete.`) so the run *looks* finished, but no further seeds/arms are ever driven — the classic "hangs at Evaluating complete, but seed_0 has results.json" symptom. Always write `(( _running++ )) || true` and `(( _running-- )) || true`. Any new `exp*.sh` must mirror `exp4_garden.sh`: guarded counters **plus** a per-seed resume guard (`[ -f .../results.json ] && skip`) so a re-run never recomputes a completed seed. Fixed in `2b2e2d9` (garden) and `1598f84` (bungeenerf).
- **Control level shifts with B_total — use `--control_level_max` on sweeps.** `set_control_level()` picks the deepest octree level where `B_total / N_active >= rho_min`. When sweeping B_total (e.g. Pareto ×0.25…×2), a larger B_total satisfies `rho_min` at deeper levels where 1x could not — so 1x and 2x end up with different control grids. A finer grid means more cells, smaller per-cell budgets, and slower per-step growth → the 2x arm underfills its budget while 1x fills ~99%. For any `exp*.sh` that sweeps B_total, pass `--control_level_max <level>` to lock the spatial partition scale across arms. Use `--control_level <N>` to force a specific level outright (-1 = auto).

### Skills

These skills are sanctioned for this project; invoke the matching one for the situation rather than improvising:

- `superpowers:brainstorming` — before any feature, component, or behavior-change work, to pin down intent and design.
- `superpowers:writing-plans` / `superpowers:executing-plans` — turn a settled design into a multi-step plan and execute it with review checkpoints.
- `tdd` (`superpowers:test-driven-development`) — author the failing test case first, then the minimal implementation. Per the Development workflow above, only run tests with no environment-conflict risk; hand any GPU/CUDA/server-dependent test to the user to run.
- `superpowers:requesting-code-review` / `code-review` — for the self-review pass and when preparing work for the user's review.
- `diagnose` (`superpowers:systematic-debugging`) — for hard bugs, test failures, or performance regressions, before proposing a fix.
- `grill-with-docs` — to stress-test a plan against CONTEXT.md and the ADRs, and update domain docs inline.
- `to-prd` / `to-issues` / `triage` — to capture a PRD, break a plan into issues, and run the issue workflow (see the issue-tracker and triage-labels sections above).
