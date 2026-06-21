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

from controller import TemporalBudgetController, ReallocationPlan
from partition import OctreePartition


def _requires_gm():
    if not _GAUSSIAN_MODEL_IMPORT_OK:
        pytest.skip("GaussianModel import requires CUDA environment")


# ---------------------------------------------------------------------------
# controller_active gate logic (server-only via GaussianModel)
# ---------------------------------------------------------------------------

class TestControllerActive:
    def _mk_gm(self, progressive=True, B_total=1000, controller_enabled=True):
        _requires_gm()
        gm = GaussianModel(B_total=B_total)
        gm.voxel_size = 1.0
        gm.levels = 4
        gm.init_level = 1
        gm.coarse_intervals = [500, 1500, 3000]
        gm._level = torch.tensor([[1]], device='cuda')
        gm.progressive = progressive
        gm._controller_enabled = controller_enabled
        gm._controller_update_from = 1500
        gm._controller_update_until = 25000
        return gm

    def test_pre_unlock_progressive_returns_false(self):
        gm = self._mk_gm(progressive=True)
        assert gm.controller_active(300) is False
        assert gm.controller_active(500) is False

    def test_post_unlock_progressive_returns_true(self):
        gm = self._mk_gm(progressive=True)
        assert gm.controller_active(3001) is True

    def test_post_update_until_returns_false(self):
        gm = self._mk_gm(progressive=True)
        assert gm.controller_active(25001) is False

    def test_B_total_zero_returns_false(self):
        gm = self._mk_gm(progressive=True, B_total=0)
        assert gm.controller_active(4000) is False

    def test_B_total_negative_returns_false(self):
        gm = self._mk_gm(progressive=True, B_total=-1)
        assert gm.controller_active(4000) is False

    def test_controller_disabled_returns_false(self):
        gm = self._mk_gm(progressive=True, controller_enabled=False)
        assert gm.controller_active(4000) is False

    def test_nonprogressive_post_update_from_returns_true(self):
        gm = self._mk_gm(progressive=False)
        gm._controller_update_from = 1500
        assert gm.controller_active(1500) is False
        assert gm.controller_active(1501) is True

    def test_nonprogressive_pre_update_from_returns_false(self):
        gm = self._mk_gm(progressive=False)
        gm._controller_update_from = 1500
        assert gm.controller_active(1400) is False


# ---------------------------------------------------------------------------
# _lowest_sa_in_surplus with GC exclusion (server-only via GaussianModel)
# ---------------------------------------------------------------------------

class TestLowestSAInSurplusWithGCExclusion:
    def test_dead_anchor_with_inf_sa_not_selected(self):
        _requires_gm()
        N = 5
        s_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device='cuda')
        anchor_cell_ids = torch.tensor([10, 10, 10, 10, 10], device='cuda')
        cell_ids = torch.tensor([10], device='cuda')
        delta = torch.tensor([-3], device='cuda')
        c_target = torch.tensor([2], device='cuda')
        plan = ReallocationPlan(cell_ids=cell_ids, delta=delta,
                                phase="steady", c_target=c_target)

        s_a_clean = s_a.clone()
        s_a_clean[0] = float('inf')
        s_a_clean[1] = float('inf')
        mask = GaussianModel._lowest_sa_in_surplus(plan, s_a_clean, anchor_cell_ids)

        expected = torch.tensor([False, False, True, True, True], device='cuda')
        assert torch.equal(mask, expected), (
            f"Dead anchors (inf s_a) should never be selected.\n"
            f"Expected {expected.tolist()}, got {mask.tolist()}"
        )



class TestOpacityDeadMaskFourParams:
    def test_accepts_four_params(self):
        _requires_gm()
        N = 2
        opacity_accum = torch.tensor([[1.0], [0.001]], device='cuda')
        anchor_demon = torch.tensor([[100.0], [100.0]], device='cuda')
        result = GaussianModel._opacity_dead_mask(
            opacity_accum, anchor_demon, 0.005, 80.0
        )
        expected = torch.tensor([False, True], device='cuda')
        assert torch.equal(result, expected)


# ---------------------------------------------------------------------------
# _cap_keep_mask — per-cell top-by-gradient cap (server-only via GaussianModel)
# ---------------------------------------------------------------------------

class TestCapKeepMask:
    def test_deficit_cell_keeps_top_by_gradient(self):
        _requires_gm()
        cell_ids = torch.tensor([10, 10, 10, 10], device='cuda')
        candidate_grads = torch.tensor([0.1, 0.9, 0.3, 0.7], device='cuda')
        plan_cell_ids = torch.tensor([10], device='cuda')
        plan_delta = torch.tensor([2], device='cuda')

        keep_mask = GaussianModel._cap_keep_mask(
            cell_ids, candidate_grads, plan_cell_ids, plan_delta)

        expected = torch.tensor([False, True, False, True], device='cuda')
        assert torch.equal(keep_mask, expected), (
            f"Should keep top 2 by gradient. Expected {expected.tolist()}, "
            f"got {keep_mask.tolist()}"
        )

    def test_surplus_cell_discards_all(self):
        _requires_gm()
        cell_ids = torch.tensor([10, 10, 10], device='cuda')
        candidate_grads = torch.tensor([0.9, 0.8, 0.7], device='cuda')
        plan_cell_ids = torch.tensor([10, 20], device='cuda')
        plan_delta = torch.tensor([-3, 2], device='cuda')

        keep_mask = GaussianModel._cap_keep_mask(
            cell_ids, candidate_grads, plan_cell_ids, plan_delta)

        assert keep_mask.sum().item() == 0, (
            f"Surplus cell (delta=-3) should keep no candidates. "
            f"Got {keep_mask.sum().item()} kept"
        )

    def test_mixed_deficit_surplus(self):
        _requires_gm()
        cell_ids = torch.tensor([10, 10, 10, 20, 20], device='cuda')
        candidate_grads = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.8], device='cuda')
        plan_cell_ids = torch.tensor([10, 20], device='cuda')
        plan_delta = torch.tensor([1, -1], device='cuda')

        keep_mask = GaussianModel._cap_keep_mask(
            cell_ids, candidate_grads, plan_cell_ids, plan_delta)

        expected = torch.tensor([False, False, True, False, False], device='cuda')
        assert torch.equal(keep_mask, expected), (
            f"Cell 10 (delta=+1) should keep top 1, cell 20 (delta=-1) keeps 0. "
            f"Expected {expected.tolist()}, got {keep_mask.tolist()}"
        )


# ---------------------------------------------------------------------------
# gather + register row-sync test (server-only — needs CUDA model)
# ---------------------------------------------------------------------------

class TestGatherRegisterRowSync:
    def test_gather_register_row_counts_synchronized(self):
        _requires_gm()
        gm = GaussianModel(B_total=100)
        gm.voxel_size = 1.0
        gm.levels = 2
        gm.init_pos = torch.tensor([0.0, 0.0, 0.0], device='cuda')
        gm.fork = 2
        gm.n_offsets = 5
        gm.feat_dim = 32
        gm.progressive = False
        gm.coarse_intervals = []

        N = 5
        gm._anchor = torch.rand(N, 3, device='cuda')
        gm._level = torch.zeros(N, 1, device='cuda')
        gm._offset = torch.zeros(N, 5, 3, device='cuda')
        gm._scaling = torch.ones(N, 6, device='cuda')
        gm._rotation = torch.zeros(N, 4, device='cuda')
        gm._rotation[:, 0] = 1.0
        gm._anchor_feat = torch.zeros(N, 32, device='cuda')
        gm._opacity = torch.zeros(N, 1, device='cuda')
        gm._extra_level = torch.zeros(N, device='cuda')

        gm.opacity_accum = torch.zeros(N, 1, device='cuda')
        gm.anchor_demon = torch.zeros(N, 1, device='cuda')
        gm.offset_denom = torch.ones(N * 5, 1, device='cuda')
        gm.offset_gradient_accum = 0.002 * torch.ones(N * 5, 1, device='cuda')

        from torch import nn
        gm.optimizer = torch.optim.Adam([
            {'params': [gm._anchor], 'lr': 0.0, 'name': 'anchor'},
            {'params': [gm._offset], 'lr': 0.0, 'name': 'offset'},
            {'params': [gm._anchor_feat], 'lr': 0.0, 'name': 'anchor_feat'},
            {'params': [gm._opacity], 'lr': 0.0, 'name': 'opacity'},
            {'params': [gm._scaling], 'lr': 0.0, 'name': 'scaling'},
            {'params': [gm._rotation], 'lr': 0.0, 'name': 'rotation'},
        ], lr=0.0)

        gm.init_pos = torch.tensor([0.0, 0.0, 0.0], device='cuda')

        grads = gm.offset_gradient_accum / gm.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (gm.offset_denom > 50.0).squeeze(dim=1)

        candidates, new_level, new_extra, cand_grads = gm._anchor_growing_gather(
            5000, grads_norm, 0.0002, 0.5, 4.0, 0.25, offset_mask)

        if candidates["anchor"].shape[0] > 0:
            gm._anchor_growing_register(candidates, new_level, new_extra)

        row_counts = {}
        for tensor_name in ['_anchor', '_offset', '_anchor_feat', '_opacity',
                             '_scaling', '_rotation']:
            t = getattr(gm, tensor_name)
            row_counts[tensor_name] = t.shape[0]

        for tensor_name in ['_level', '_extra_level', 'opacity_accum',
                             'anchor_demon']:
            t = getattr(gm, tensor_name)
            row_counts[tensor_name] = t.shape[0]

        row_counts['offset_denom'] = gm.offset_denom.shape[0]
        row_counts['offset_gradient_accum'] = gm.offset_gradient_accum.shape[0]

        N_anchor = row_counts['_anchor']
        n_offsets = gm.n_offsets
        assert row_counts['_offset'] == N_anchor, (
            f"_offset rows {row_counts['_offset']} != _anchor rows {N_anchor}"
        )
        assert row_counts['_anchor_feat'] == N_anchor
        assert row_counts['_opacity'] == N_anchor
        assert row_counts['_scaling'] == N_anchor
        assert row_counts['_rotation'] == N_anchor
        assert row_counts['_level'] == N_anchor
        assert row_counts['_extra_level'] == N_anchor
        assert row_counts['opacity_accum'] == N_anchor
        assert row_counts['anchor_demon'] == N_anchor
        assert row_counts['offset_denom'] == N_anchor * n_offsets, (
            f"offset_denom rows {row_counts['offset_denom']} != "
            f"{N_anchor} * {n_offsets}"
        )
        assert row_counts['offset_gradient_accum'] == N_anchor * n_offsets

        for group in gm.optimizer.param_groups:
            if 'mlp' in group.get('name', '') or 'embedding' in group.get('name', ''):
                continue
            p = group['params'][0]
            assert p.shape[0] == row_counts.get(group['name'], p.shape[0]), (
                f"optimizer param '{group['name']}' shape[0]={p.shape[0]} "
                f"!= expected {row_counts.get(group['name'])}"
            )


# ---------------------------------------------------------------------------
# cap logic (local — pure tensor test on TemporalBudgetController)
# ---------------------------------------------------------------------------

class TestAnchorGrowingCappedLogic:
    def test_cap_keeps_top_by_gradient(self):
        cell_ids = torch.tensor([100, 200], dtype=torch.long)
        d_A = torch.tensor([10.0, 30.0], dtype=torch.float32)
        occupancy = torch.tensor([5, 3], dtype=torch.long)
        B_total = 20

        bc = TemporalBudgetController()
        plan = bc.plan(cell_ids, d_A, occupancy, B_total)
        delta_100 = plan.delta[plan.cell_ids == 100].item()
        delta_200 = plan.delta[plan.cell_ids == 200].item()
        assert delta_100 > 0 or delta_200 > 0, "at least one deficit cell expected"

    def test_non_deficit_cells_get_no_candidates(self):
        bc = TemporalBudgetController()
        cell_ids = torch.tensor([10], dtype=torch.long)
        d = torch.tensor([5.0], dtype=torch.float32)
        n = torch.tensor([30], dtype=torch.long)
        plan = bc.plan(cell_ids, d, n, B_total=20)
        delta_0 = plan.delta[0].item()
        assert delta_0 <= 0, "surplus cell should not grow"


# ---------------------------------------------------------------------------
# partition control_level one-time guard logic (local)
# ---------------------------------------------------------------------------

class TestSetControlLevelOneShot:
    def test_first_call_succeeds(self):
        p = OctreePartition(
            B_total=100, floor=1, rho_min=2, A_min=2,
            voxel_size=1.0, fork=2, levels=4,
            init_pos=torch.tensor([0.0, 0.0, 0.0])
        )
        positions = torch.tensor([
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [0.1, 0.6, 0.1],
            [0.6, 0.1, 0.6],
        ])
        level = p.set_control_level(positions)
        assert isinstance(level, int)
        assert 0 <= level < 4

    def test_second_call_raises(self):
        p = OctreePartition(
            B_total=100, floor=1, rho_min=2, A_min=2,
            voxel_size=1.0, fork=2, levels=4,
            init_pos=torch.tensor([0.0, 0.0, 0.0])
        )
        positions = torch.tensor([
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [0.1, 0.6, 0.1],
            [0.6, 0.1, 0.6],
        ])
        p.set_control_level(positions)
        with pytest.raises(RuntimeError, match="already set"):
            p.set_control_level(positions)


# ---------------------------------------------------------------------------
# steady-phase demand-prune integration (server-only — runs adjust_anchor on
# real anchor state with the controller forced into steady phase)
# ---------------------------------------------------------------------------

class _StubProducer:
    """DemandProducer that returns a fixed s_a (lets the test craft the
    per-anchor demand ranking that drives prune-by-s(a))."""

    def __init__(self, s_a):
        self._s_a = s_a

    def produce(self, scene, stats):
        return self._s_a


class _SteadyStubController:
    """Controller forced into steady phase with a surplus plan: target one
    anchor per cell, so delta = 1 - occupancy (< 0 wherever a cell holds more
    than one anchor). Isolates the actuator's demand-prune execution from the
    controller's water-fill (which is unit-tested separately)."""

    def plan(self, cell_ids, d_A, occupancy, B_total, d_B=None):
        c_target = torch.ones_like(occupancy)
        delta = c_target - occupancy
        return ReallocationPlan(cell_ids=cell_ids, delta=delta,
                                phase="steady", c_target=c_target)

    def reset(self):
        pass

    @property
    def reached_phase2(self):
        return True


class TestSteadyDemandPruneIntegration:
    """End-to-end adjust_anchor in steady phase: demand-prune executes,
    executed <= planned, total <= B_total.

    Grow is suppressed (offset_gradient_accum = 0 -> no candidates -> weed_out
    never runs, so no camera setup is needed); this isolates the prune path
    that the ramp-only training smoke never exercises.
    """

    def test_demand_prune_executes_and_respects_budget(self):
        _requires_gm()
        gm = GaussianModel(B_total=1000)
        gm.voxel_size = 1.0
        gm.fork = 2
        gm.levels = 2
        gm.n_offsets = 5
        gm.feat_dim = 32
        gm.progressive = False
        gm.coarse_intervals = []
        gm.init_pos = torch.tensor([0.0, 0.0, 0.0], device='cuda')

        # 4 anchors in cell A (origin), 2 in cell B (far away) -> 2 control cells.
        positions = torch.tensor([
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0], [100.0, 100.0, 100.0],
        ], device='cuda')
        N = positions.shape[0]
        k = gm.n_offsets

        gm._anchor = positions.clone()
        gm._level = torch.zeros(N, 1, device='cuda')
        gm._offset = torch.zeros(N, k, 3, device='cuda')
        gm._scaling = torch.ones(N, 6, device='cuda')
        gm._rotation = torch.zeros(N, 4, device='cuda')
        gm._rotation[:, 0] = 1.0
        gm._anchor_feat = torch.zeros(N, gm.feat_dim, device='cuda')
        gm._opacity = torch.zeros(N, 1, device='cuda')
        gm._extra_level = torch.zeros(N, device='cuda')

        # Healthy opacity -> GC mask all False (mean opacity 1.0 >> min_opacity).
        gm.opacity_accum = torch.ones(N, 1, device='cuda')
        gm.anchor_demon = torch.ones(N, 1, device='cuda')
        # Zero gradients -> no grow candidates -> weed_out skipped entirely.
        gm.offset_denom = torch.ones(N * k, 1, device='cuda')
        gm.offset_gradient_accum = torch.zeros(N * k, 1, device='cuda')

        gm.optimizer = torch.optim.Adam([
            {'params': [gm._anchor], 'lr': 0.0, 'name': 'anchor'},
            {'params': [gm._offset], 'lr': 0.0, 'name': 'offset'},
            {'params': [gm._anchor_feat], 'lr': 0.0, 'name': 'anchor_feat'},
            {'params': [gm._opacity], 'lr': 0.0, 'name': 'opacity'},
            {'params': [gm._scaling], 'lr': 0.0, 'name': 'scaling'},
            {'params': [gm._rotation], 'lr': 0.0, 'name': 'rotation'},
        ], lr=0.0)

        gm.partition = OctreePartition(
            B_total=gm.B_total, floor=1, rho_min=1, A_min=1,
            voxel_size=gm.voxel_size, fork=gm.fork, levels=gm.levels,
            init_pos=gm.init_pos,
        )
        gm.partition.set_control_level(gm._anchor)
        gm._control_level_set = True

        # Distinct s_a within each cell -> deterministic lowest-s(a) pruning.
        # Cell A (idx 0-3): [1,2,3,4] keep 4.  Cell B (idx 4-5): [5,6] keep 6.
        s_a = torch.tensor([1., 2., 3., 4., 5., 6.], device='cuda')
        gm.demand_producer = _StubProducer(s_a)
        gm.controller = _SteadyStubController()

        gm._controller_enabled = True
        gm._controller_update_from = 0
        gm._controller_update_until = 10_000

        assert torch.unique(gm.partition.cell_id(gm._anchor)).numel() == 2, \
            "fixture must form exactly 2 control cells"

        n_before = gm.get_anchor.shape[0]
        # adjust_anchor mutates optimizer-registered leaf params in place;
        # training always calls it under no_grad (train.py), so mirror that.
        with torch.no_grad():
            gm.adjust_anchor(iteration=100)
        n_after = gm.get_anchor.shape[0]

        # Planned prune = sum |delta| over surplus cells = (4-1) + (2-1) = 4.
        planned_prune = (4 - 1) + (2 - 1)
        executed_prune = n_before - n_after

        assert executed_prune <= planned_prune, (
            f"executed prune {executed_prune} exceeds planned {planned_prune}")
        assert n_after <= gm.B_total, f"total {n_after} > B_total {gm.B_total}"
        # Demand-prune fired: each cell collapses to its c_target = 1.
        assert n_after == 2, f"expected 2 survivors (1 per cell), got {n_after}"
        assert torch.unique(gm.partition.cell_id(gm.get_anchor)).numel() == 2, \
            "expected exactly one survivor per cell"
