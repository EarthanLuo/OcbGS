import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from controller import StaticBudgetController, ReallocationPlan, BudgetController


class TestThreePartInvariant:
    def test_binding_scenario(self):
        bc = StaticBudgetController()
        B_total = 100
        cell_ids = torch.arange(5, dtype=torch.long)
        d = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        occupancy = torch.tensor([8, 13, 18, 30, 31], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.delta.sum().item() == 0, "invariant 1: Σδ = 0"
        assert plan.c_target.sum().item() <= B_total, "invariant 2: Σc* ≤ B_total"
        assert plan.c_target.sum().item() == B_total, "invariant 3: binding ⇒ Σc* ≡ B_total"


class TestUniformBudgetSteady:
    def test_equal_demand_yields_equal_targets(self):
        bc = StaticBudgetController()
        B_total = 100
        N = 4
        cell_ids = torch.arange(N, dtype=torch.long)
        d = torch.ones(N)
        occupancy = torch.tensor([25, 25, 25, 25], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        unique_targets = torch.unique(plan.c_target)
        assert unique_targets.numel() == 1
        assert unique_targets.item() == 25


class TestUniformRamp:
    def test_equal_demand_proportional_fill_no_negative_delta(self):
        bc = StaticBudgetController()
        B_total = 100
        N = 4
        cell_ids = torch.arange(N, dtype=torch.long)
        d = torch.ones(N)
        occupancy = torch.tensor([10, 15, 20, 25], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="ramp")

        assert (plan.delta >= 0).all(), "ramp: δ ≥ 0 for all cells"
        assert plan.delta.sum().item() > 0, "ramp: should grow toward B_total"
        assert (plan.c_target >= occupancy).all(), "ramp: c* ≥ n for all cells"


class TestSkewed:
    def test_high_demand_gets_above_mean(self):
        bc = StaticBudgetController()
        B_total = 500
        cell_ids = torch.arange(6, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
        occupancy = torch.tensor([50, 50, 50, 50, 50, 100], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        c_target = plan.c_target
        mean = c_target.float().mean().item()
        assert c_target[5].item() > mean, "high-demand cell gets > mean"
        assert c_target[0].item() < mean, "low-demand cell gets < mean"


class TestCapBinds:
    def test_high_demand_clipped_at_cap_residual_redistributed(self):
        bc = StaticBudgetController(k_cap=2)
        B_total = 100
        cell_ids = torch.arange(5, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0, 96.0])
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        m = B_total / 5
        cap = min(2 * m, 0.25 * B_total)
        assert plan.c_target[4].item() <= cap + 1
        assert plan.c_target.sum().item() == B_total
        assert (plan.c_target[:4] > 1).all(), "unclamped cells received redistributed residual"


class TestFloorBinds:
    def test_low_demand_held_at_floor(self):
        bc = StaticBudgetController(floor=5)
        B_total = 100
        cell_ids = torch.arange(4, dtype=torch.long)
        d = torch.tensor([0.001, 1.0, 1.0, 1.0])
        occupancy = torch.tensor([1, 33, 33, 33], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.c_target[0].item() >= 5, "floor applied to active (occupied) cell"

    def test_floor_not_applied_to_empty_inactive_cell(self):
        bc = StaticBudgetController(floor=5)
        B_total = 100
        cell_ids = torch.arange(3, dtype=torch.long)
        d = torch.tensor([0.0, 1.0, 1.0])
        occupancy = torch.tensor([0, 50, 50], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.c_target[0].item() == 0, "inactive cell (n=0,d=0) gets no floor"


class TestRateLimitBinds:
    def test_sum_abs_delta_capped_at_rate_limit(self):
        bc = StaticBudgetController(rate_limit=0.05)
        B_total = 100
        cell_ids = torch.arange(5, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        occupancy = torch.tensor([0, 0, 100, 0, 0], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        limit = int(0.05 * B_total)
        assert plan.delta.abs().sum().item() <= limit

    def test_proportional_scaling_preserved_when_sigma_delta_zero(self):
        bc = StaticBudgetController(rate_limit=0.05)
        B_total = 320
        cell_ids = torch.arange(4, dtype=torch.long)
        d = torch.ones(4)
        occupancy = torch.tensor([40, 40, 120, 120], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        limit = int(0.05 * B_total)
        assert plan.delta.abs().sum().item() <= limit
        assert plan.delta.abs().sum().item() > 0, "not collapsed to zero"
        assert torch.equal(plan.delta, torch.tensor([4, 4, -4, -4], dtype=torch.long))


class TestDeadBandBinds:
    def test_small_delta_zeroed_step7_restores_balance(self):
        bc = StaticBudgetController(theta_frac=0.25)
        B_total = 200
        cell_ids = torch.arange(4, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0])
        occupancy = torch.tensor([20, 30, 60, 90], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.delta.sum().item() == 0, "step 7 must restore Σδ = 0"


class TestMultipleConstraints:
    def test_cap_and_rate_limit_both_active(self):
        bc = StaticBudgetController(k_cap=2, rate_limit=0.05)
        B_total = 200
        cell_ids = torch.arange(6, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 95.0])
        occupancy = torch.tensor([10, 10, 100, 10, 10, 60], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.delta.sum().item() == 0, "step 7 restores balance under combined constraints"
        assert plan.delta.abs().sum().item() <= int(0.05 * B_total), "rate-limit holds"
        assert plan.c_target.sum().item() <= B_total, "Σc* ≤ B_total"


class TestIntegerExactness:
    def test_hamilton_produces_exact_sum(self):
        bc = StaticBudgetController()
        B_total = 97
        cell_ids = torch.arange(5, dtype=torch.long)
        d = torch.tensor([3.0, 7.0, 11.0, 13.0, 17.0])
        occupancy = torch.tensor([5, 10, 20, 25, 37], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.c_target.sum().item() == B_total
        assert plan.c_target.dtype == torch.long

    def test_hamilton_tie_break_deterministic(self):
        bc = StaticBudgetController()
        B_total = 102
        cell_ids = torch.arange(6, dtype=torch.long)
        d = torch.ones(6)
        occupancy = torch.tensor([17, 17, 17, 17, 17, 17], dtype=torch.long)

        plan1 = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")
        plan2 = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert torch.equal(plan1.c_target, plan2.c_target)
        assert torch.equal(plan1.delta, plan2.delta)


class TestFloorSumExceedsBTotal:
    def test_raises_value_error(self):
        bc = StaticBudgetController(floor=10)
        B_total = 50
        cell_ids = torch.arange(8, dtype=torch.long)
        d = torch.ones(8)
        occupancy = torch.ones(8, dtype=torch.long)

        with pytest.raises(ValueError, match="floor"):
            bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")


class TestEmptyNActive:
    def test_zero_active_returns_empty_plan(self):
        bc = StaticBudgetController()
        cell_ids = torch.empty(0, dtype=torch.long)
        d = torch.empty(0)
        occupancy = torch.empty(0, dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total=100, phase="steady")

        assert plan.cell_ids.numel() == 0
        assert plan.delta.numel() == 0
        assert plan.c_target.numel() == 0


class TestRampProportionalClamp:
    def test_proportional_clamp_lands_on_B_total(self):
        bc = StaticBudgetController()
        B_total = 100
        cell_ids = torch.arange(4, dtype=torch.long)
        d = torch.tensor([1.0, 1.0, 1.0, 1.0])
        occupancy = torch.tensor([0, 0, 35, 25], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="ramp")

        assert (plan.delta >= 0).all(), "ramp: no pruning"
        new_total = occupancy.sum().item() + plan.delta.sum().item()
        assert new_total == B_total, "ramp clamp lands exactly on B_total"

    def test_headroom_less_than_grow_cells_lands_exactly(self):
        bc = StaticBudgetController()
        B_total = 11
        cell_ids = torch.arange(10, dtype=torch.long)
        d = torch.ones(10)
        occupancy = torch.zeros(10, dtype=torch.long)
        occupancy[0] = 10

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="ramp")

        assert (plan.delta >= 0).all(), "ramp: no pruning"
        new_total = occupancy.sum().item() + plan.delta.sum().item()
        assert new_total == B_total, "lands exactly on B_total even with fractional p·δ"


class TestDeterminism:
    def test_same_inputs_same_outputs(self):
        bc = StaticBudgetController()
        B_total = 100
        cell_ids = torch.arange(6, dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        occupancy = torch.tensor([10, 15, 20, 10, 25, 20], dtype=torch.long)

        plan1 = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")
        plan2 = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")
        plan3 = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert torch.equal(plan1.cell_ids, plan3.cell_ids)
        assert torch.equal(plan1.delta, plan3.delta)
        assert torch.equal(plan2.delta, plan1.delta)
        assert torch.equal(plan1.c_target, plan2.c_target)
        assert plan1.phase == plan2.phase


class TestCapUndershoot:
    def test_all_cells_at_cap_sum_less_than_B_total(self):
        bc = StaticBudgetController(k_cap=1)
        B_total = 100
        cell_ids = torch.arange(2, dtype=torch.long)
        d = torch.tensor([1.0, 1.0])
        occupancy = torch.tensor([20, 20], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total, phase="steady")

        assert plan.c_target.sum().item() < B_total, "undershoot: Σc* < B_total"
        assert plan.c_target.sum().item() > 0, "still allocates something"


class TestExtremeBTotal:
    def test_B_total_zero_handled_gracefully(self):
        bc = StaticBudgetController()
        cell_ids = torch.arange(3, dtype=torch.long)
        d = torch.tensor([1.0, 2.0, 3.0])
        occupancy = torch.tensor([10, 10, 10], dtype=torch.long)

        plan = bc._allocate(cell_ids, d, occupancy, B_total=0, phase="steady")

        assert plan.c_target.sum().item() == 0
        assert plan.delta.sum().item() <= 0

    def test_B_total_one_with_N_active_two_raises_floor_error(self):
        bc = StaticBudgetController(floor=1)
        cell_ids = torch.arange(2, dtype=torch.long)
        d = torch.tensor([1.0, 1.0])
        occupancy = torch.tensor([10, 10], dtype=torch.long)

        with pytest.raises(ValueError, match="floor"):
            bc._allocate(cell_ids, d, occupancy, B_total=1, phase="steady")


def test_ramp_keeps_small_deficit_growth():
    """ramp dead-band bug regression: deficit < thr should NOT be zeroed.
    m=10 → thr=0.25*10=2.5. occ=9 → deficit=1 < 2.5 → currently zeroed (bug).
    After fix (dead-band only in steady), delta=1 survives → Σn grows."""
    bc = StaticBudgetController(floor=1, k_cap=8)
    B_total, C = 100, 10
    cell_ids = torch.arange(C, dtype=torch.long)
    d = torch.ones(C)
    occupancy = torch.full((C,), 9, dtype=torch.long)
    plan = bc.plan(cell_ids, d, occupancy, B_total)
    grown = (occupancy + plan.delta).sum().item()
    assert grown == pytest.approx(B_total, abs=1), (
        f"ramp dead-banded a real deficit: "
        f"\u03a3n={grown} stuck below B_total={B_total}"
    )


class TestSubclassCompliance:
    def test_is_budget_controller(self):
        bc = StaticBudgetController()
        assert isinstance(bc, BudgetController)

    def test_plan_signature_matches_abc(self):
        import inspect
        sig = inspect.signature(BudgetController.plan)
        impl_sig = inspect.signature(StaticBudgetController.plan)
        for name in sig.parameters:
            assert name in impl_sig.parameters
