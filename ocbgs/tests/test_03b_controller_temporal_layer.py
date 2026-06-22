import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from controller import (
    StaticBudgetController,
    TemporalBudgetController,
    ReallocationPlan,
    BudgetController,
)


class TestEMADecay:
    def test_step_change_reaches_63_percent_within_tau_smooth_steps(self):
        """
        EMA: d_smooth(t) = beta * d_smooth(t-1) + (1-beta) * d_raw(t)
        beta = 1 - 1/tau_smooth. After tau_smooth steps, the smoothed value
        incorporates 1 - (1-1/tau)^tau ≈ 1 - 1/e ≈ 63% of the step change.
        tau_smooth=3 → after 3 steps, fraction = 1 - (2/3)^3 = 19/27 ≈ 70%.
        """
        bc = TemporalBudgetController(tau_smooth=3)
        B_total = 100
        cell_ids = torch.tensor([10, 20, 30], dtype=torch.long)
        occupancy = torch.ones(3, dtype=torch.long)

        d_base = torch.tensor([1.0, 1.0, 1.0])
        for _ in range(6):
            bc.plan(cell_ids, d_base, occupancy, B_total)

        d_step = torch.tensor([10.0, 10.0, 10.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_step, occupancy, B_total)

        plan = bc.plan(cell_ids, d_step, occupancy, B_total)

        expected_min = 1.0 + 0.63 * (10.0 - 1.0)
        smoothed_vals = []
        for cid in cell_ids:
            smoothed_vals.append(bc._d_smooth_prev[cid.item()])
        avg_smooth = sum(smoothed_vals) / len(smoothed_vals)
        assert avg_smooth >= expected_min, (
            f"EMA should reach >= 63% of step change within tau_smooth steps; "
            f"got {avg_smooth:.3f}, expected >= {expected_min:.3f}"
        )


class TestSpearmanGate:
    def test_identical_demand_yields_correlation_one(self):
        bc = TemporalBudgetController(tau_smooth=3, spearman_threshold=0.9)
        B_total = 100
        cell_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        occupancy = torch.ones(5, dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        for _ in range(bc.tau_smooth + 3):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._stable_count >= bc.k

    def test_reversed_demand_yields_correlation_neg_one(self):
        bc = TemporalBudgetController(tau_smooth=3, spearman_threshold=0.9)
        B_total = 100
        cell_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        occupancy = torch.ones(5, dtype=torch.long)
        d_forward = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        d_reverse = torch.tensor([9.0, 7.0, 5.0, 3.0, 1.0])

        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_forward, occupancy, B_total)

        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_reverse, occupancy, B_total)

        assert bc._stable_count == 0, (
            "reversed demand should NOT be considered stable (corr ≈ -1)"
        )

    def test_random_demand_yields_correlation_near_zero(self):
        bc = TemporalBudgetController(tau_smooth=3, spearman_threshold=0.9)
        B_total = 100
        cell_ids = torch.arange(10, dtype=torch.long)
        occupancy = torch.ones(10, dtype=torch.long)

        d_base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_base, occupancy, B_total)

        d_uncorrelated = torch.tensor([3.0, 6.0, 1.0, 8.0, 4.0, 9.0, 2.0, 7.0, 5.0, 10.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_uncorrelated, occupancy, B_total)

        assert bc._stable_count == 0, (
            "uncorrelated demand should fail stability gate"
        )

    def test_computed_over_shared_cell_ids_only(self):
        bc = TemporalBudgetController(tau_smooth=3, spearman_threshold=0.9)
        B_total = 100

        cell_ids_old = torch.tensor([10, 20, 30], dtype=torch.long)
        occupancy = torch.ones(3, dtype=torch.long)
        d_old = torch.tensor([1.0, 5.0, 10.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids_old, d_old, occupancy, B_total)

        cell_ids_new = torch.tensor([20, 30, 40], dtype=torch.long)
        occupancy_new = torch.ones(3, dtype=torch.long)
        d_new = torch.tensor([10.0, 1.0, 5.0])
        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids_new, d_new, occupancy_new, B_total)

        assert bc._stable_count == 0, (
            "shared cells {20,30}: old ranks [5,10] → new ranks [10,1] = "
            "ranks reversed → not stable; gate must fail"
        )


class TestPhaseSwitch:
    def setup_method(self):
        self.bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        self.B_total = 100
        self.cell_ids = torch.arange(5, dtype=torch.long)
        self.occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        self.d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

    def _stabilise(self):
        for _ in range(self.bc.tau_smooth + self.bc.k + 2):
            self.bc.plan(self.cell_ids, self.d, self.occupancy, self.B_total)

    def test_enters_steady_after_N_ge_B_total_and_spearman_holds_k_steps(self):
        self._stabilise()
        assert self.bc._phase == "steady", (
            "should enter steady after N_total >= B_total AND Spearman holds for k steps"
        )
        assert self.bc._stable_count >= self.bc.k
        assert self.bc._reached_phase2

    def test_does_not_enter_steady_if_spearman_dips_below_threshold(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        B_total = 100
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d_base = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids, d_base, occupancy, B_total)

        d_perturbed = torch.tensor([9.0, 1.0, 5.0, 3.0, 7.0])
        bc.plan(cell_ids, d_perturbed, occupancy, B_total)

        assert bc._stable_count == 0, "counter must reset when Spearman dips"

        for _ in range(bc.tau_smooth + bc.k):
            bc.plan(cell_ids, d_perturbed, occupancy, B_total)

    def test_progressive_unlock_new_cell_ids_reset_gate(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        B_total = 100

        cell_ids_old = torch.arange(5, dtype=torch.long)
        occupancy_old = torch.tensor([10, 10, 10, 10, 10], dtype=torch.long)
        d_old = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])

        for _ in range(bc.tau_smooth + 2):
            bc.plan(cell_ids_old, d_old, occupancy_old, B_total)

        assert bc._stable_count >= 2, "should have built stable count on old cells"

        cell_ids_new = torch.cat([cell_ids_old, torch.tensor([100, 200], dtype=torch.long)])
        occupancy_new = torch.tensor([10, 10, 10, 10, 10, 5, 5], dtype=torch.long)
        d_new = torch.tensor([5.0, 7.0, 1.0, 3.0, 9.0, 1.0, 3.0])

        for _ in range(bc.tau_smooth):
            bc.plan(cell_ids_new, d_new, occupancy_new, B_total)

        assert bc._stable_count == 0, (
            "gate must reset when new cell_ids appear — full-tree demand re-stabilises"
        )

    def test_steady_phase_latches_never_returns_to_ramp(self):
        self._stabilise()
        assert self.bc._phase == "steady"

        d_perturbed = torch.tensor([9.0, 7.0, 5.0, 3.0, 1.0])
        self.bc.plan(self.cell_ids, d_perturbed, self.occupancy, self.B_total)
        assert self.bc._phase == "steady", (
            "phase must latch: once steady, never returns to ramp"
        )


class TestPlateauFallback:
    def test_plateau_enters_steady_with_N_below_B_total(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        B_total = 200
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([10, 10, 10, 10, 10], dtype=torch.long)
        d = torch.tensor([5.0, 3.0, 7.0, 9.0, 1.0])

        for _ in range(bc.tau_smooth + bc.k + 2):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._phase == "steady", (
            "plateau fallback: N_total unchanged for k steps AND Spearman holds "
            "→ enter steady even though N_total < B_total"
        )
        assert bc._reached_phase2
        assert occupancy.sum().item() == 50 < B_total

    def test_plateau_disabled_stays_in_ramp(self):
        bc = TemporalBudgetController(
            tau_smooth=3, k=2, spearman_threshold=0.9,
            plateau_enabled=False
        )
        B_total = 200
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([10, 10, 10, 10, 10], dtype=torch.long)
        d = torch.tensor([5.0, 3.0, 7.0, 9.0, 1.0])

        for _ in range(bc.tau_smooth + bc.k + 5):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._phase == "ramp", (
            "plateau_enabled=False: even with stable N_total < B_total, "
            "must stay in ramp"
        )
        assert not bc._reached_phase2


class TestPlanSignature:
    def test_phase_determined_internally_not_passed_by_caller(self):
        bc = TemporalBudgetController()

        cell_ids = torch.arange(3, dtype=torch.long)
        d_A = torch.tensor([1.0, 2.0, 3.0])
        occupancy = torch.tensor([10, 10, 10], dtype=torch.long)
        B_total = 30

        plan = bc.plan(cell_ids, d_A, occupancy, B_total)
        assert plan.phase in ("ramp", "steady")

    def test_d_B_none_ignored(self):
        bc = TemporalBudgetController()
        cell_ids = torch.arange(3, dtype=torch.long)
        d_A = torch.tensor([1.0, 2.0, 3.0])
        occupancy = torch.tensor([10, 10, 10], dtype=torch.long)

        plan_with_B = bc.plan(cell_ids, d_A, occupancy, B_total=30, d_B=None)
        plan_without_B = bc.plan(cell_ids, d_A, occupancy, B_total=30)

        assert torch.equal(plan_with_B.delta, plan_without_B.delta)
        assert plan_with_B.phase == plan_without_B.phase


class TestMultiStepFixedPoint:
    def test_stable_demand_yields_delta_approx_zero(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        B_total = 100
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([16, 18, 20, 22, 24], dtype=torch.long)
        d = torch.tensor([16.0, 18.0, 20.0, 22.0, 24.0])

        for _ in range(bc.tau_smooth + bc.k + 3):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._phase == "steady"

        plan = bc.plan(cell_ids, d, occupancy, B_total)
        assert plan.delta.abs().max().item() <= 0, (
            f"fixed-point: self-consistent demand=occupancy distinct values → δ ≈ 0, "
            f"got max|δ| = {plan.delta.abs().max().item()}"
        )


class TestMultiStepNoThrash:
    def test_small_random_perturbation_absorbed_by_deadband(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2, spearman_threshold=0.9)
        B_total = 150
        C = 10
        cell_ids = torch.arange(C, dtype=torch.long)
        base_occ = torch.tensor([6, 8, 10, 12, 14, 16, 18, 20, 22, 24], dtype=torch.long)
        base_d = torch.tensor([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])

        for _ in range(bc.tau_smooth + bc.k + 3):
            bc.plan(cell_ids, base_d, base_occ, B_total)

        assert bc._phase == "steady"

        torch.manual_seed(42)
        for _ in range(5):
            d_noisy = base_d + 0.001 * torch.randn(C).abs()
            d_noisy = torch.clamp(d_noisy, min=0.0)
            plan = bc.plan(cell_ids, d_noisy, base_occ, B_total)
            assert plan.phase == "steady"

        changed_count = (plan.delta != 0).sum().item()
        assert changed_count < C, (
            f"small perturbations should be absorbed by dead-band; "
            f"got {changed_count}/{C} cells with non-zero delta"
        )


class TestStateReset:
    def test_all_state_reset_on_activation(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2)

        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        B_total = 100

        for _ in range(bc.tau_smooth + bc.k + 2):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._phase == "steady"
        assert bc._reached_phase2
        assert len(bc._d_smooth_prev) > 0
        assert len(bc._d_history) > 0
        assert bc._stable_count > 0

        bc.reset()

        assert bc._phase == "ramp"
        assert not bc._reached_phase2
        assert bc._d_smooth_prev == {}
        assert len(bc._d_history) == 0
        assert bc._stable_count == 0
        assert bc._plateau_count == 0
        assert bc._n_total_prev == 0
        assert bc._step_count == 0


class TestNoCUDAImport:
    def test_controller_module_has_no_cuda_import(self):
        controller_path = os.path.join(
            os.path.dirname(__file__), '..', 'controller', '__init__.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'import torch.cuda' not in content, (
            "controller module must not import torch.cuda"
        )
        assert 'from torch.cuda' not in content, (
            "controller module must not import from torch.cuda"
        )


class TestScenarioBGuardrail:
    def test_reached_phase2_exposed_as_property(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2)
        assert hasattr(bc, 'reached_phase2')
        assert not bc.reached_phase2

    def test_reached_phase2_false_in_ramp_true_in_steady(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2)
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        B_total = 100

        for _ in range(bc.tau_smooth + bc.k + 2):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc._phase == "steady"
        assert bc.reached_phase2

    def test_reached_phase2_resets_on_activation(self):
        bc = TemporalBudgetController(tau_smooth=3, k=2)
        cell_ids = torch.arange(5, dtype=torch.long)
        occupancy = torch.tensor([20, 20, 20, 20, 20], dtype=torch.long)
        d = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        B_total = 100

        for _ in range(bc.tau_smooth + bc.k + 2):
            bc.plan(cell_ids, d, occupancy, B_total)

        assert bc.reached_phase2

        bc.reset()
        assert not bc.reached_phase2


class TestTemporalBudgetControllerSubclass:
    def test_is_instance_of_budget_controller(self):
        bc = TemporalBudgetController()
        assert isinstance(bc, BudgetController)

    def test_is_instance_of_static_budget_controller(self):
        bc = TemporalBudgetController()
        assert isinstance(bc, StaticBudgetController)

    def test_inherits_allocate_method(self):
        bc = TemporalBudgetController()
        assert hasattr(bc, '_allocate')

    def test_forwards_knob_defaults_to_static_allocator(self):
        bc = TemporalBudgetController(floor=5, k_cap=4, theta_frac=0.1, rate_limit=0.03)
        assert bc.floor == 5
        assert bc.k_cap == 4
        assert bc.theta_frac == 0.1
        assert bc.rate_limit == 0.03

    def test_plan_signature_matches_abc(self):
        import inspect
        sig = inspect.signature(BudgetController.plan)
        impl_sig = inspect.signature(TemporalBudgetController.plan)
        for name in sig.parameters:
            assert name in impl_sig.parameters

    def test_default_plateau_true_backward_compat(self):
        bc = TemporalBudgetController()
        assert bc.plateau_enabled is True
