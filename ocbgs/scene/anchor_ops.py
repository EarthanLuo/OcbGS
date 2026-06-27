"""Pure-tensor anchor operations, factored out of GaussianModel so they can be
unit-tested on CPU without the CUDA-only imports (torch_scatter, simple_knn).
"""
import torch


def cap_keep_mask(cell_ids, candidate_grads, plan_cell_ids, plan_delta):
    """Keep the top-``delta`` candidates by gradient within each plan cell.

    Vectorized replacement for the original O(C) Python loop (which did a
    per-cell ``.item()``/scan/top-k over C≈85k cells × N≈360k candidates —
    the dominant cost of adjust_anchor under relax). Semantics: for every plan
    cell with ``delta > 0``, keep the ``min(delta, n_cand)`` highest-gradient
    candidates falling in that cell; all other candidates are dropped.

    Args:
        cell_ids: Tensor[N] cell id of each candidate anchor.
        candidate_grads: Tensor[N] per-candidate gradient (the ranking key).
        plan_cell_ids: Tensor[C] unique cell ids from the controller plan.
        plan_delta: Tensor[C] per-cell growth budget (may be ≤ 0).

    Returns:
        Tensor[N] bool keep mask.
    """
    N = cell_ids.shape[0]
    device = cell_ids.device
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if N == 0:
        return keep_mask

    pos = plan_delta > 0
    if not bool(pos.any()):
        return keep_mask
    pos_cells = plan_cell_ids[pos]
    pos_delta = plan_delta[pos].to(torch.long)

    # Map each candidate to its cell's budget via searchsorted on sorted plan cells.
    order = torch.argsort(pos_cells)
    sorted_cells = pos_cells[order].contiguous()
    sorted_delta = pos_delta[order]
    idx = torch.searchsorted(sorted_cells, cell_ids).clamp(max=sorted_cells.shape[0] - 1)
    matched = sorted_cells[idx] == cell_ids
    cand_delta = torch.where(matched, sorted_delta[idx], torch.zeros_like(sorted_delta[idx]))

    active = cand_delta > 0
    if not bool(active.any()):
        return keep_mask

    a_cell = cell_ids[active]
    a_grad = candidate_grads[active]
    a_idx = torch.where(active)[0]
    a_delta = cand_delta[active]

    # Order candidates grouped by cell (asc), gradient descending within each cell.
    g_order = torch.argsort(a_grad, descending=True, stable=True)
    c_order = torch.argsort(a_cell[g_order], stable=True)
    final = g_order[c_order]

    sorted_cells2 = a_cell[final]
    M = sorted_cells2.shape[0]
    arange = torch.arange(M, device=device)
    new_group = torch.ones(M, dtype=torch.bool, device=device)
    if M > 1:
        new_group[1:] = sorted_cells2[1:] != sorted_cells2[:-1]
    # Forward-fill each group's start index via cummax → within-cell rank.
    seg_start = torch.where(new_group, arange, torch.zeros_like(arange))
    seg_start = torch.cummax(seg_start, dim=0)[0]
    rank_within = arange - seg_start

    keep_sorted = rank_within < a_delta[final]
    keep_mask[a_idx[final][keep_sorted]] = True
    return keep_mask
