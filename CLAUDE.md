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
