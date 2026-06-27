"""Equivalence test: vectorized cap_keep_mask must match the original
per-cell Python-loop top-k semantics (keep the top-delta candidates by
gradient within each plan cell with positive delta).

Loads scene/anchor_ops.py directly by path so the test runs on CPU without
the CUDA-only imports (torch_scatter, simple_knn) that scene.gaussian_model
pulls in.
"""
import importlib.util
import os

import torch

_anchor_ops_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scene", "anchor_ops.py",
)
_spec = importlib.util.spec_from_file_location("anchor_ops", _anchor_ops_path)
anchor_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anchor_ops)
cap_keep_mask = anchor_ops.cap_keep_mask


def _ref_cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta):
    """The original O(C) Python loop — reference semantics."""
    N = cell_ids.shape[0]
    device = cell_ids.device
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    for i in range(plan_cell_ids.shape[0]):
        cid = plan_cell_ids[i].item()
        delta = plan_delta[i].item()
        if delta <= 0:
            continue
        in_cell = (cell_ids == cid)
        n_cand = int(in_cell.sum().item())
        if n_cand == 0:
            continue
        k = min(int(delta), n_cand)
        if k <= 0:
            continue
        g_in_cell = candidate_grads[in_cell]
        _, topk = torch.topk(g_in_cell, k, largest=True)
        cell_indices = torch.where(in_cell)[0]
        keep_mask[cell_indices[topk]] = True
    return keep_mask


def _distinct_grads(n, seed):
    """Strictly-distinct grads so top-k has no tie ambiguity."""
    g = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).float() + 1.0
    return g / n


def test_matches_reference_random():
    torch.manual_seed(0)
    N = 500
    cell_ids = torch.randint(0, 20, (N,))
    candidate_grads = _distinct_grads(N, 1)
    plan_cell_ids = torch.arange(20)
    plan_delta = torch.randint(-2, 8, (20,))

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    ref = _ref_cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert torch.equal(got, ref)


def test_delta_exceeds_cell_count_keeps_all_in_cell():
    cell_ids = torch.tensor([5, 5, 5, 9])
    candidate_grads = torch.tensor([0.3, 0.1, 0.2, 0.9])
    plan_cell_ids = torch.tensor([5, 9])
    plan_delta = torch.tensor([10, 1])  # cell 5 budget 10 > 3 candidates → keep all 3

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    ref = _ref_cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert torch.equal(got, ref)
    assert got.tolist() == [True, True, True, True]


def test_candidate_cell_absent_from_plan_excluded():
    cell_ids = torch.tensor([1, 2, 3])      # cell 3 not in plan
    candidate_grads = torch.tensor([0.5, 0.6, 0.7])
    plan_cell_ids = torch.tensor([1, 2])
    plan_delta = torch.tensor([1, 1])

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    ref = _ref_cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert torch.equal(got, ref)
    assert bool(got[2].item()) is False


def test_all_deltas_nonpositive_keeps_nothing():
    cell_ids = torch.tensor([1, 1, 2])
    candidate_grads = torch.tensor([0.5, 0.6, 0.7])
    plan_cell_ids = torch.tensor([1, 2])
    plan_delta = torch.tensor([0, -3])

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert bool(got.any().item()) is False


def test_empty_candidates():
    cell_ids = torch.zeros(0, dtype=torch.long)
    candidate_grads = torch.zeros(0)
    plan_cell_ids = torch.tensor([1, 2])
    plan_delta = torch.tensor([3, 4])

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert got.shape[0] == 0


def test_keeps_highest_grad_within_cell():
    # cell 7 has 4 candidates, budget 2 → keep the two highest grads (0.9, 0.8)
    cell_ids = torch.tensor([7, 7, 7, 7])
    candidate_grads = torch.tensor([0.9, 0.1, 0.8, 0.2])
    plan_cell_ids = torch.tensor([7])
    plan_delta = torch.tensor([2])

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert got.tolist() == [True, False, True, False]


def test_large_random_matches_reference():
    torch.manual_seed(7)
    N = 5000
    cell_ids = torch.randint(0, 300, (N,))
    candidate_grads = _distinct_grads(N, 2)
    plan_cell_ids = torch.arange(300)
    plan_delta = torch.randint(-1, 20, (300,))

    got = cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    ref = _ref_cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta)
    assert torch.equal(got, ref)
