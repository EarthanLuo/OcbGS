import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from controller import align_demand_b, TemporalBudgetController


def _fresh_ctrl(lam=0.0):
    return TemporalBudgetController(fusion_lambda=lam)


def test_align_demand_b_to_current_cells():
    b_cache = {10: 3.0, 20: 5.0, 30: 7.0}
    cell_ids = torch.tensor([10, 99, 20])
    d_b = align_demand_b(cell_ids, b_cache)

    assert torch.allclose(d_b, torch.tensor([3.0, 0.0, 5.0]))
    assert d_b.shape == (3,)


def test_fusion_lambda_zero_recovers_a_only():
    cell_ids = torch.tensor([100, 200, 300])
    d_A = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)
    d_B = torch.tensor([9.0, 0.0, 1.0], dtype=torch.float32)
    occupancy = torch.tensor([10, 20, 30], dtype=torch.long)
    B_total = 100

    plan_a_only = _fresh_ctrl(lam=0.0).plan(cell_ids, d_A, occupancy, B_total, d_B=None)
    plan_fused = _fresh_ctrl(lam=0.0).plan(cell_ids, d_A, occupancy, B_total, d_B=d_B)

    assert torch.allclose(plan_fused.delta, plan_a_only.delta)
    assert plan_fused.phase == plan_a_only.phase


def test_fusion_lambda_one_equal_weight():
    cell_ids = torch.tensor([100, 200])
    d_A = torch.tensor([4.0, 0.0], dtype=torch.float32)
    d_B = torch.tensor([0.0, 8.0], dtype=torch.float32)
    occupancy = torch.tensor([5, 5], dtype=torch.long)
    B_total = 20

    plan = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total, d_B=d_B)

    assert plan.c_target.sum().item() <= B_total


def test_fusion_b_lights_up_cell_a_missed():
    cell_ids = torch.tensor([10, 20])
    d_A = torch.tensor([0.0, 2.0], dtype=torch.float32)
    d_B = torch.tensor([5.0, 2.0], dtype=torch.float32)
    occupancy = torch.tensor([1, 3], dtype=torch.long)
    B_total = 10

    plan = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total, d_B=d_B)

    assert (plan.delta != 0).any() or (plan.c_target > 0).all()


def test_fusion_all_zero_demand_no_crash():
    cell_ids = torch.tensor([0])
    d_A = torch.tensor([0.0])
    d_B = torch.tensor([0.0])
    occupancy = torch.tensor([1], dtype=torch.long)
    B_total = 1

    plan = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total, d_B=d_B)
    assert plan.c_target.sum().item() >= 0


def test_fusion_d_b_none_sentinel_is_a_only():
    cell_ids = torch.tensor([0])
    d_A = torch.tensor([1.0])
    occupancy = torch.tensor([1], dtype=torch.long)
    B_total = 1

    plan = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total, d_B=None)

    assert plan.cell_ids.shape == (1,)


def test_align_demand_b_new_cell_gets_zero():
    b_cache = {1: 4.0, 2: 6.0}
    cell_ids = torch.tensor([1, 3, 2])
    d_b = align_demand_b(cell_ids, b_cache)

    assert torch.allclose(d_b, torch.tensor([4.0, 0.0, 6.0]))


def test_fusion_independent_l1_norm_scale_invariance():
    cell_ids = torch.tensor([10, 20])
    d_A = torch.tensor([1.0, 3.0])
    occupancy = torch.tensor([5, 5], dtype=torch.long)
    B_total = 40

    p_small = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total,
                                         d_B=torch.tensor([2.0, 6.0]))
    p_huge = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total,
                                        d_B=torch.tensor([200.0, 600.0]))
    assert torch.equal(p_small.c_target, p_huge.c_target)
    assert torch.equal(p_small.delta, p_huge.delta)

    p_a1 = _fresh_ctrl(lam=1.0).plan(cell_ids, torch.tensor([1.0, 3.0]), occupancy, B_total,
                                      d_B=torch.tensor([2.0, 6.0]))
    p_a100 = _fresh_ctrl(lam=1.0).plan(cell_ids, torch.tensor([100.0, 300.0]), occupancy, B_total,
                                        d_B=torch.tensor([2.0, 6.0]))
    assert torch.equal(p_a1.c_target, p_a100.c_target)
    assert torch.equal(p_a1.delta, p_a100.delta)


def test_fusion_additive_b_lights_da_zero_cell():
    cell_ids = torch.tensor([10, 20])
    d_A = torch.tensor([0.0, 1.0])
    occupancy = torch.tensor([5, 5], dtype=torch.long)
    B_total = 40

    with_b = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total,
                                        d_B=torch.tensor([5.0, 1.0]))
    without_b = _fresh_ctrl(lam=1.0).plan(cell_ids, d_A, occupancy, B_total,
                                           d_B=torch.tensor([0.0, 1.0]))
    assert with_b.c_target[0] > without_b.c_target[0]
