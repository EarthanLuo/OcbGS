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

### Skills

These skills are sanctioned for this project; invoke the matching one for the situation rather than improvising:

- `superpowers:brainstorming` — before any feature, component, or behavior-change work, to pin down intent and design.
- `superpowers:writing-plans` / `superpowers:executing-plans` — turn a settled design into a multi-step plan and execute it with review checkpoints.
- `tdd` (`superpowers:test-driven-development`) — author the failing test case first, then the minimal implementation. Per the Development workflow above, only run tests with no environment-conflict risk; hand any GPU/CUDA/server-dependent test to the user to run.
- `superpowers:requesting-code-review` / `code-review` — for the self-review pass and when preparing work for the user's review.
- `diagnose` (`superpowers:systematic-debugging`) — for hard bugs, test failures, or performance regressions, before proposing a fix.
- `grill-with-docs` — to stress-test a plan against CONTEXT.md and the ADRs, and update domain docs inline.
- `to-prd` / `to-issues` / `triage` — to capture a PRD, break a plan into issues, and run the issue workflow (see the issue-tracker and triage-labels sections above).
