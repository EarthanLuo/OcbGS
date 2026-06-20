from abc import ABC, abstractmethod
import torch


class Partition(ABC):

    @abstractmethod
    def set_control_level(self, anchor_positions):
        """
        Derive the Control Level from anchor positions and Capacity Budget.

        Returns:
            control_level: int, the octree level at which Control Cells are formed
        """
        ...

    @abstractmethod
    def cell_id(self, anchor_positions):
        """
        Assign each anchor to exactly one Control Cell (Cell Membership).

        Returns:
            Tensor[N] of int cell indices
        """
        ...

    @abstractmethod
    def reduce(self, anchor_positions, weights, exclude=None):
        """
        Reduce per-anchor weights to per-Control-Cell Demand Scores d(v).

        Args:
            anchor_positions: Tensor[N, 3] anchor positions
            weights: Tensor[N] per-anchor weights (s(a))
            exclude: optional mask of anchors to exclude

        Returns:
            cell_ids: Tensor[C] unique cell indices
            d_v: Tensor[C] per-cell demand scores
        """
        ...


class StubPartition(Partition):

    def set_control_level(self, anchor_positions):
        return 0

    def cell_id(self, anchor_positions):
        return torch.zeros(anchor_positions.shape[0], dtype=torch.long, device=anchor_positions.device)

    def reduce(self, anchor_positions, weights, exclude=None):
        if weights.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=weights.device),
                torch.empty(0, dtype=torch.float, device=weights.device),
            )
        d_v = weights.sum().unsqueeze(0)
        cell_ids = torch.zeros(1, dtype=torch.long, device=weights.device)
        return cell_ids, d_v
