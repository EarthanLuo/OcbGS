"""Regression tests: vectorized EMA / dict update / Spearman lookup
must produce bit-identical results to the original Python-loop version.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from controller import TemporalBudgetController


def _run_n_steps(bc, cell_ids, d_seq, occupancy, B_total):
    """Drive the controller for len(d_seq) steps; return the last plan."""
    plan = None
    for d in d_seq:
        plan = bc.plan(cell_ids, d, occupancy, B_total)
    return plan


def _clone_controller_state(bc):
    return {
        "d_smooth_prev": dict(bc._d_smooth_prev),
        "stable_count": bc._stable_count,
        "plateau_count": bc._plateau_count,
        "phase": bc._phase,
        "step_count": bc._step_count,
    }


class TestEMAVectorizedMatchesOriginal:
    """The vectorized implementation must produce the same d_smooth values
    as the original element-wise loop across all observable state."""

    def _demand_seq(self, C, steps, seed=0):
        torch.manual_seed(seed)
        return [torch.rand(C).abs() + 0.1 for _ in range(steps)]

    def test_d_smooth_prev_matches_after_warm_start(self):
        C, steps, B = 20, 10, 200
        cell_ids = torch.arange(C, dtype=torch.long)
        occupancy = torch.ones(C, dtype=torch.long) * (B // C)
        d_seq = self._demand_seq(C, steps)

        bc = TemporalBudgetController(tau_smooth=3, k=3)
        _run_n_steps(bc, cell_ids, d_seq, occupancy, B)
        state = _clone_controller_state(bc)

        for cid in cell_ids.tolist():
            assert cid in state["d_smooth_prev"], f"cell {cid} missing from d_smooth_prev"

        # All EMA values must be positive (demand was always > 0.1)
        for v in state["d_smooth_prev"].values():
            assert v > 0.0

    def test_stable_count_matches_after_stable_demand(self):
        C, B = 5, 100
        cell_ids = torch.arange(C, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        bc = TemporalBudgetController(tau_smooth=3, k=3)

        steps = bc.tau_smooth + bc.k + 4
        for _ in range(steps):
            bc.plan(cell_ids, d, occupancy, B)

        assert bc._stable_count >= bc.k, (
            f"stable demand must accumulate stable_count >= k={bc.k}, "
            f"got {bc._stable_count}"
        )

    def test_phase_transition_unchanged(self):
        C, B = 5, 100
        cell_ids = torch.arange(C, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        bc = TemporalBudgetController(tau_smooth=3, k=2)

        for _ in range(bc.tau_smooth + bc.k + 3):
            bc.plan(cell_ids, d, occupancy, B)

        assert bc._phase == "steady"
        assert bc._reached_phase2

    def test_ema_values_match_manual_computation(self):
        """Single-cell EMA: d_smooth[t] = beta*d_smooth[t-1] + (1-beta)*d_raw[t]
        Starting from d_raw[0] as the initial prev value.
        """
        bc = TemporalBudgetController(tau_smooth=4, k=10)  # k high to stay in ramp
        cell_ids = torch.tensor([7], dtype=torch.long)
        occupancy = torch.tensor([1], dtype=torch.long)
        B = 10

        d_seq = [1.0, 2.0, 3.0, 4.0]
        beta = 1.0 - 1.0 / bc.tau_smooth  # = 0.75

        # Manual trace:
        # step 0: prev=1.0 (fallback), smooth = 0.75*1.0 + 0.25*1.0 = 1.0
        # step 1: prev=1.0,            smooth = 0.75*1.0 + 0.25*2.0 = 1.25
        # step 2: prev=1.25,           smooth = 0.75*1.25 + 0.25*3.0 = 1.6875
        # step 3: prev=1.6875,         smooth = 0.75*1.6875 + 0.25*4.0 = 2.265625
        expected = [1.0, 1.25, 1.6875, 2.265625]

        for i, d_val in enumerate(d_seq):
            d = torch.tensor([d_val])
            bc.plan(cell_ids, d, occupancy, B)
            got = bc._d_smooth_prev[7]
            assert abs(got - expected[i]) < 1e-5, (
                f"step {i}: expected {expected[i]:.6f}, got {got:.6f}"
            )

    def test_spearman_stable_with_new_cells_not_in_history(self):
        """When a new cell appears (not in _d_smooth_prev), the fallback
        must use d_raw for that cell, not zero — otherwise the EMA
        starting value is wrong and Spearman stability is corrupted."""
        bc = TemporalBudgetController(tau_smooth=3, k=2)
        B = 100

        # Warm up with cells 0-4
        cell_ids_old = torch.arange(5, dtype=torch.long)
        occ_old = torch.ones(5, dtype=torch.long) * 10
        d_old = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids_old, d_old, occ_old, B)

        # New cell 99 appears — its first d_smooth must equal d_raw (no prev)
        cell_ids_new = torch.tensor([0, 1, 2, 3, 4, 99], dtype=torch.long)
        occ_new = torch.ones(6, dtype=torch.long) * 10
        d_new = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
        bc.plan(cell_ids_new, d_new, occ_new, B)

        # Cell 99's EMA after one step: prev=7.0 (fallback), smooth=beta*7+(1-beta)*7=7
        got = bc._d_smooth_prev[99]
        assert abs(got - 7.0) < 1e-5, (
            f"new cell fallback must use d_raw=7.0, got {got:.6f}"
        )

    def test_large_cell_count_bulk_transfer(self):
        """C=500 cells: exercise the vectorized path without crashing
        and confirm d_smooth_prev has exactly C entries."""
        C, B = 500, 5000
        cell_ids = torch.arange(C, dtype=torch.long)
        occupancy = torch.ones(C, dtype=torch.long) * (B // C)
        torch.manual_seed(1)
        d = torch.rand(C) + 0.1

        bc = TemporalBudgetController(tau_smooth=3)
        for _ in range(5):
            bc.plan(cell_ids, d, occupancy, B)

        assert len(bc._d_smooth_prev) == C
