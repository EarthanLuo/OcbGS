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


class StubBudgetController(BudgetController):

    def plan(self, cell_ids, d_A, occupancy, B_total, d_B=None):
        C = cell_ids.shape[0]
        return ReallocationPlan(
            cell_ids=cell_ids,
            delta=torch.zeros(C, dtype=torch.long, device=cell_ids.device),
            phase="ramp",
            c_target=occupancy.clone(),
        )
