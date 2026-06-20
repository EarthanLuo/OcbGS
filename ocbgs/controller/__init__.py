from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ReallocationPlan:
    cell_ids: torch.Tensor
    delta: torch.Tensor
    phase: str
    c_target: torch.Tensor


class BudgetController(ABC):

    @abstractmethod
    def plan(self, cell_ids, d_A, occupancy, B_total, d_B=None):
        """
        Compute a ReallocationPlan from demand and occupancy.

        Args:
            cell_ids: Tensor[C] unique Control Cell indices
            d_A: Tensor[C] Demand Scores d(v) from Source A
            occupancy: Tensor[C] Cell Occupancy n(v)
            B_total: int, Capacity Budget (upper bound on total anchors)
            d_B: Optional[Tensor[C]], Demand Scores from Source B (issue 06)

        Returns:
            ReallocationPlan with cell_ids, delta, phase, c_target
        """
        ...


class StaticBudgetController(BudgetController):
    def __init__(self, floor=1, k_cap=8, theta_frac=0.25, rate_limit=0.05):
        self.floor = floor
        self.k_cap = k_cap
        self.theta_frac = theta_frac
        self.rate_limit = rate_limit

    def plan(self, cell_ids, d_A, occupancy, B_total, d_B=None):
        return self._allocate(cell_ids, d_A, occupancy, B_total, phase="ramp")

    def _allocate(self, cell_ids, d, occupancy, B_total, phase):
        C = cell_ids.shape[0]
        device = cell_ids.device

        if C == 0:
            return ReallocationPlan(
                cell_ids=cell_ids,
                delta=torch.empty(0, dtype=torch.long, device=device),
                phase=phase,
                c_target=torch.empty(0, dtype=torch.long, device=device),
            )

        if B_total <= 0:
            return ReallocationPlan(
                cell_ids=cell_ids,
                delta=torch.zeros(C, dtype=torch.long, device=device),
                phase=phase,
                c_target=torch.zeros(C, dtype=torch.long, device=device),
            )

        n = occupancy
        active = (n > 0) | (d > 0)
        N_active = active.sum().item()

        if N_active == 0:
            return ReallocationPlan(
                cell_ids=cell_ids,
                delta=torch.zeros(C, dtype=torch.long, device=device),
                phase=phase,
                c_target=torch.zeros(C, dtype=torch.long, device=device),
            )

        if self.floor * N_active > B_total:
            raise ValueError(
                f"\u03a3floor ({self.floor * N_active}) > B_total ({B_total}); "
                f"control_level derivation should preclude this"
            )

        d_sum = d.sum().item()
        if d_sum <= 0:
            t = torch.full((C,), B_total / N_active, dtype=torch.float32, device=device)
        else:
            t = B_total * d / d_sum

        m = B_total / N_active
        cap_val = min(self.k_cap * m, 0.25 * B_total)

        max_iter = 100
        tol = 1e-4
        for _ in range(max_iter):
            t = torch.where(active, torch.clamp(t, min=self.floor), t)
            t = torch.clamp(t, max=cap_val)

            t_sum = t.sum().item()
            residual = B_total - t_sum
            if abs(residual) < tol:
                break

            if residual > 0:
                unclamped = t < (cap_val - tol)
            else:
                unclamped = active & (t > (self.floor + tol))

            if unclamped.sum().item() == 0:
                break

            unclamped_d = d[unclamped]
            unclamped_d_sum = unclamped_d.sum().item()
            if unclamped_d_sum <= 0:
                break

            t[unclamped] += residual * (unclamped_d / unclamped_d_sum)

        t_sum = t.sum().item()
        total_alloc = int(round(t_sum))
        total_alloc = min(total_alloc, B_total)

        floor_t = torch.floor(t).long()
        R = total_alloc - floor_t.sum().item()

        if R > 0:
            remainders = t - floor_t.float()
            _, indices = torch.sort(remainders, descending=True, stable=True)
            bonus = torch.zeros(C, dtype=torch.long, device=device)
            bonus[indices[:R]] = 1
            c_target = floor_t + bonus
        else:
            c_target = floor_t

        delta = c_target - n

        thr = torch.maximum(
            torch.tensor(1.0, device=device),
            self.theta_frac * c_target.float()
        )
        zero_mask = delta.abs().float() < thr
        delta = torch.where(zero_mask, torch.zeros_like(delta), delta)

        if phase == "steady":
            sum_abs = delta.abs().sum().item()
            limit = self.rate_limit * B_total
            if sum_abs > limit:
                scale = limit / sum_abs
                delta = torch.trunc(delta.float() * scale).long()

        if phase == "steady":
            net = delta.sum().item()
            if net > 0:
                pos_mask = delta > 0
                pos_indices = torch.where(pos_mask)[0]
                pos_vals = delta[pos_mask]
                if pos_vals.numel() > 0:
                    sorted_order = torch.argsort(pos_vals, stable=True)
                    sorted_vals = pos_vals[sorted_order]
                    sorted_indices = pos_indices[sorted_order]

                    cumsum = torch.cumsum(sorted_vals, dim=0)
                    ge_mask = cumsum >= net
                    if ge_mask.any():
                        j = torch.where(ge_mask)[0][0].item()
                    else:
                        j = sorted_vals.numel()

                    for i in range(j):
                        delta[sorted_indices[i]] = 0

                    if j < sorted_vals.numel():
                        prev = cumsum[j - 1].item() if j > 0 else 0
                        delta[sorted_indices[j]] -= (net - prev)

            elif net < 0:
                neg_net = -net
                neg_mask = delta < 0
                neg_indices = torch.where(neg_mask)[0]
                neg_abs = -delta[neg_mask]
                if neg_abs.numel() > 0:
                    sorted_order = torch.argsort(neg_abs, stable=True)
                    sorted_vals = neg_abs[sorted_order]
                    sorted_indices = neg_indices[sorted_order]

                    cumsum = torch.cumsum(sorted_vals, dim=0)
                    ge_mask = cumsum >= neg_net
                    if ge_mask.any():
                        j = torch.where(ge_mask)[0][0].item()
                    else:
                        j = sorted_vals.numel()

                    for i in range(j):
                        delta[sorted_indices[i]] = 0

                    if j < sorted_vals.numel():
                        prev = cumsum[j - 1].item() if j > 0 else 0
                        delta[sorted_indices[j]] += (neg_net - prev)

        if phase == "ramp":
            delta = torch.clamp(delta, min=0)
            N_total = n.sum().item()
            sum_delta = delta.sum().item()

            if N_total >= B_total:
                delta.zero_()
            elif N_total + sum_delta > B_total:
                p = (B_total - N_total) / sum_delta
                delta_f = delta.float() * p
                delta = torch.floor(delta_f).long()

                allocated = N_total + delta.sum().item()
                remaining = B_total - allocated
                if remaining > 0:
                    remainders = delta_f - delta.float()
                    valid = delta_f > 0
                    adj_rem = torch.where(
                        valid, remainders,
                        torch.tensor(-1e9, device=device)
                    )
                    _, top_idx = torch.sort(adj_rem, descending=True, stable=True)
                    give = min(remaining, valid.sum().item())
                    if give > 0:
                        delta[top_idx[:give]] += 1

        return ReallocationPlan(
            cell_ids=cell_ids,
            delta=delta,
            phase=phase,
            c_target=c_target,
        )


class StubBudgetController(BudgetController):

    def plan(self, cell_ids, d_A, occupancy, B_total, d_B=None):
        C = cell_ids.shape[0]
        return ReallocationPlan(
            cell_ids=cell_ids,
            delta=torch.zeros(C, dtype=torch.long, device=cell_ids.device),
            phase="ramp",
            c_target=occupancy.clone(),
        )
