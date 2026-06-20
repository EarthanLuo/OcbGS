import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

try:
    from scene.gaussian_model import GaussianModel
    _GAUSSIAN_MODEL_IMPORT_OK = True
except ImportError:
    _GAUSSIAN_MODEL_IMPORT_OK = False

from controller import ReallocationPlan


def _requires_gm():
    if not _GAUSSIAN_MODEL_IMPORT_OK:
        pytest.skip("GaussianModel import requires CUDA environment")


class TestOpacityDeadMask:
    def test_opacity_below_threshold_and_mature_becomes_true(self):
        _requires_gm()
        N = 5
        opacity_accum = torch.tensor([[1.0], [0.5], [0.001], [2.0], [0.0]])
        anchor_demon = torch.tensor([[120.0], [90.0], [85.0], [50.0], [100.0]])
        min_opacity = 0.005
        maturity_min = 80.0

        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, min_opacity, maturity_min
        )

        expected_dead = torch.tensor([False, False, True, False, False])
        assert torch.equal(mask, expected_dead), (
            f"opacity_dead_mask: expected {expected_dead.tolist()}, got {mask.tolist()}\n"
            f"Mean opacities: {opacity_accum.flatten() / anchor_demon.flatten()}"
        )

    def test_opacity_below_but_not_mature_becomes_false(self):
        _requires_gm()
        N = 3
        opacity_accum = torch.tensor([[0.001], [0.001], [0.001]])
        anchor_demon = torch.tensor([[50.0], [79.0], [80.0]])
        min_opacity = 0.005
        maturity_min = 80.0

        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, min_opacity, maturity_min
        )

        assert mask[0].item() == False, "not mature"
        assert mask[1].item() == False, "not mature (strict)"
        assert mask[2].item() == True, "mature AND dead → True"

    def test_zero_anchors_returns_empty_mask(self):
        _requires_gm()
        opacity_accum = torch.empty(0, 1)
        anchor_demon = torch.empty(0, 1)
        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, 0.005, 80.0
        )
        assert mask.numel() == 0
        assert mask.dtype == torch.bool

    def test_all_dead_all_mature_returns_all_true(self):
        _requires_gm()
        N = 4
        opacity_accum = torch.zeros(N, 1)
        anchor_demon = torch.full((N, 1), 100.0)
        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, 0.005, 80.0
        )
        assert mask.all()
        assert mask.shape == (N,)

    def test_all_alive_returns_all_false(self):
        _requires_gm()
        N = 4
        opacity_accum = torch.full((N, 1), 10.0)
        anchor_demon = torch.full((N, 1), 100.0)
        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, 0.005, 80.0
        )
        assert not mask.any()
        assert mask.shape == (N,)

    def test_anchor_demon_zero_maturity_gate_blocks(self):
        _requires_gm()
        opacity_accum = torch.tensor([[0.0]])
        anchor_demon = torch.tensor([[0.0]])
        mask = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, 0.005, 80.0
        )
        assert not mask.item(), "anchor_demon=0 → not mature → never dead"


def _make_plan(cell_ids, delta, phase="steady"):
    C = len(cell_ids)
    return ReallocationPlan(
        cell_ids=torch.tensor(cell_ids, dtype=torch.long),
        delta=torch.tensor(delta, dtype=torch.long),
        phase=phase,
        c_target=torch.zeros(C, dtype=torch.long),
    )


class TestLowestSAInSurplus:
    def test_single_surplus_cell_selects_lowest_sa(self):
        _requires_gm()
        plan = _make_plan([10], [-3])

        N = 5
        s_a = torch.tensor([0.9, 0.1, 0.5, 0.2, 0.8])
        anchor_cell_ids = torch.tensor([10, 10, 10, 10, 10], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        expected = torch.tensor([False, True, False, True, False])
        selected = torch.where(mask)[0]
        assert torch.equal(mask, expected), (
            f"Should select anchors with lowest s(a): 0.1, 0.2; got indices {selected.tolist()}"
        )
        assert s_a[selected].sum().item() == pytest.approx(0.3)

    def test_multiple_surplus_cells_independent_selection(self):
        _requires_gm()
        plan = _make_plan([10, 20], [-2, -1])

        N = 7
        s_a = torch.tensor([0.9, 0.1, 0.5, 0.2, 0.8, 0.3, 0.7])
        anchor_cell_ids = torch.tensor([10, 10, 10, 10, 20, 20, 20], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert mask[1].item(), "anchor at idx 1 s_a=0.1 in cell 10"
        assert mask[3].item(), "anchor at idx 3 s_a=0.2 in cell 10"
        assert mask[5].item(), "anchor at idx 5 s_a=0.3 in cell 20 (only 1 to remove)"

        assert not mask[0].item(), "s_a=0.9 in cell 10, not selected (only 2 removed)"
        assert not mask[2].item(), "s_a=0.5 in cell 10, not selected"
        assert not mask[4].item(), "s_a=0.8 in cell 20, not selected (only 1 removed)"
        assert not mask[6].item(), "s_a=0.7 in cell 20, not selected"

        pruned_count = mask.sum().item()
        assert pruned_count == 3, f"should prune exactly 2+1=3, got {pruned_count}"

    def test_deficit_cell_no_anchors_selected(self):
        _requires_gm()
        plan = _make_plan([10, 20], [5, -3])

        N = 6
        s_a = torch.tensor([0.9, 0.1, 0.5, 0.2, 0.8, 0.3])
        anchor_cell_ids = torch.tensor([10, 10, 10, 20, 20, 20], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert mask[0].item() == False, "deficit cell 10 → no pruning"
        assert mask[1].item() == False, "deficit cell 10 → no pruning"
        assert mask[2].item() == False, "deficit cell 10 → no pruning"

        assert mask[3:].sum().item() == 3, (
            "surplus cell 20 with |delta|=3, has 3 anchors → all should be selected"
        )

    def test_surplus_delta_exceeds_anchors_selects_all(self):
        _requires_gm()
        plan = _make_plan([10], [-10])

        N = 3
        s_a = torch.tensor([0.5, 0.3, 0.8])
        anchor_cell_ids = torch.tensor([10, 10, 10], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert mask.all(), "|delta|=10 > n=3 → select all 3 anchors"
        assert mask.shape == (3,)

    def test_delta_zero_no_anchors_selected(self):
        _requires_gm()
        plan = _make_plan([10, 20], [3, 0])

        N = 5
        s_a = torch.tensor([0.9, 0.5, 0.8, 0.3, 0.7])
        anchor_cell_ids = torch.tensor([10, 10, 20, 20, 20], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert not mask.any(), "no surplus cell → nothing to prune"

    def test_empty_mask_with_no_surplus_cells(self):
        _requires_gm()
        plan = _make_plan([10, 20, 30], [0, 3, 5])

        N = 4
        s_a = torch.tensor([0.9, 0.5, 0.8, 0.3])
        anchor_cell_ids = torch.tensor([10, 10, 20, 30], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert not mask.any(), "no delta < 0 → empty prune set"

    def test_empty_inputs(self):
        _requires_gm()
        plan = _make_plan([], [])
        s_a = torch.empty(0)
        anchor_cell_ids = torch.empty(0, dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert mask.numel() == 0
        assert mask.dtype == torch.bool

    def test_tie_break_on_equal_sa(self):
        _requires_gm()
        plan = _make_plan([10], [-2])

        N = 4
        s_a = torch.tensor([0.5, 0.5, 0.5, 0.5])
        anchor_cell_ids = torch.tensor([10, 10, 10, 10], dtype=torch.long)

        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a, anchor_cell_ids)

        assert mask.sum().item() == 2, "should select exactly 2 even with ties"


class TestNoStateMutation:
    def test_opacity_dead_mask_no_state_access(self):
        _requires_gm()
        import inspect
        sig = inspect.signature(GaussianModel._opacity_dead_mask)
        param_names = list(sig.parameters.keys())
        assert 'self' not in param_names, "_opacity_dead_mask is staticmethod"
        assert len(param_names) == 4

    def test_lowest_sa_in_surplus_no_state_access(self):
        _requires_gm()
        import inspect
        sig = inspect.signature(GaussianModel._lowest_sa_in_surplus)
        param_names = list(sig.parameters.keys())
        assert 'self' not in param_names, "_lowest_sa_in_surplus is staticmethod"
        assert len(param_names) == 3
