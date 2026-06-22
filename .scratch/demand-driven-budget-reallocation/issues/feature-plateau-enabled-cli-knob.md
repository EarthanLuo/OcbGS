# Issue: Expose `plateau_enabled` knob to CLI

**Status:** ready-for-agent

## Context

Spec §6.3 Exp 1 defines two operating points:

- **Matched-budget:** force `Σn ≡ B_total` (plateau OFF; only enter steady when N_total >= B_total)
- **Natural-budget:** cap `Σn ≤ B_total` (plateau ON; enter steady when N_total stabilizes, even below B_total)

The plateau fallback mechanism is hardcoded in `TemporalBudgetController.plan()` (controller/__init__.py:347-352). Currently plateau is always ON — the `--no_plateau` / `--matched_budget` CLI flag does not exist.

This issue exposes the knob so Exp 1 can toggle matched vs natural from the command line.

## What to build

1. **`arguments/__init__.py`:** Add `self.plateau_enabled = True` + `--no_plateau` parser.
2. **`controller/__init__.py`:** `TemporalBudgetController.__init__` accepts `plateau_enabled=True`. In `plan()`, change `plateau_eligible` to gate on `self.plateau_enabled`:

   ```python
   plateau_eligible = (self.plateau_enabled and
                       self._plateau_count >= self.k and
                       self._stable_count >= self.k)
   ```

   **CRITICAL:** gate only `plateau_eligible`, NOT the entire phase-transition block (line 347-352). The `N_total >= B_total` path (path ①) must remain ungated.
3. **`gaussian_model.py`:** Pass `plateau_enabled=opt.plateau_enabled` at line 447.

## Acceptance criteria

- [ ] `--no_plateau` on CLI sets `plateau_enabled=False`, preventing early steady entry when N_total < B_total.
- [ ] Default (`plateau_enabled=True`) preserves existing behavior — backward compatible.
- [ ] `TemporalBudgetController(plateau_enabled=False)` + fed plateau + stable sequence → stays in "ramp".
- [ ] `TemporalBudgetController(plateau_enabled=True)` + same sequence → enters "steady" (existing behavior).
- [ ] Pure-logic unit tests only (no CUDA); runnable locally.
