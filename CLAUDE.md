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

### Skills

These skills are sanctioned for this project; invoke the matching one for the situation rather than improvising:

- `superpowers:brainstorming` — before any feature, component, or behavior-change work, to pin down intent and design.
- `superpowers:writing-plans` / `superpowers:executing-plans` — turn a settled design into a multi-step plan and execute it with review checkpoints.
- `tdd` (`superpowers:test-driven-development`) — author the failing test case first, then the minimal implementation. Per the Development workflow above, only run tests with no environment-conflict risk; hand any GPU/CUDA/server-dependent test to the user to run.
- `superpowers:requesting-code-review` / `code-review` — for the self-review pass and when preparing work for the user's review.
- `diagnose` (`superpowers:systematic-debugging`) — for hard bugs, test failures, or performance regressions, before proposing a fix.
- `grill-with-docs` — to stress-test a plan against CONTEXT.md and the ADRs, and update domain docs inline.
- `to-prd` / `to-issues` / `triage` — to capture a PRD, break a plan into issues, and run the issue workflow (see the issue-tracker and triage-labels sections above).
