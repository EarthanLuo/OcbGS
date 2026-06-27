import torch
from controller import resolve_controller_demand, StaticBudgetController


def test_resolve_returns_demand_unchanged_when_not_uniform():
    d_a = torch.tensor([3.0, 1.0, 0.0, 5.0])
    out = resolve_controller_demand(d_a, uniform=False)
    assert torch.equal(out, d_a)


def test_resolve_zeros_demand_when_uniform():
    d_a = torch.tensor([3.0, 1.0, 0.0, 5.0])
    out = resolve_controller_demand(d_a, uniform=True)
    assert torch.equal(out, torch.zeros_like(d_a))
    assert out.shape == d_a.shape


def test_zeroed_demand_drives_controller_uniform_branch():
    ctrl = StaticBudgetController(floor=1, k_cap=8)
    cell_ids = torch.arange(4)
    occupancy = torch.tensor([10, 10, 10, 10])
    d_zero = resolve_controller_demand(torch.tensor([9.0, 1.0, 1.0, 1.0]), uniform=True)
    plan = ctrl.plan(cell_ids, d_A=d_zero, occupancy=occupancy, B_total=40)
    assert plan.c_target.tolist() == [10, 10, 10, 10]
