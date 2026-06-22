from abc import ABC, abstractmethod
import torch
import warnings

# Prime multipliers for spatial hashing (FowlerNollVo-inspired).
# Collision-free for bounded scene coordinates (grid cells within ~1e6
# per axis).  A debug assert inside reduce() verifies collision-freedom
# at each call.
_HASH_P1 = 73856093
_HASH_P2 = 19349663
_HASH_P3 = 83492791


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


class OctreePartition(Partition):
    """Spatial partition into Control Cells at a derived control_level.

    Stateless except for the once-derived, frozen control_level (hence
    cell_size).  Cell Membership is recomputed each call via round-division
    and prime-multiplier spatial hashing (practically collision-free for
    bounded scene coordinates).  The segment-sum reduction inside reduce()
    uses unique(coords, dim=0) for exact grouping and includes a debug
    assertion against hash collisions.

    Parameters
    ----------
    B_total : int
        Capacity Budget (total anchor count target).
    floor : int
        Minimum anchors per Control Cell (Controller knob).
    rho_min : float
        Minimum mean occupancy per Control Cell (derivation knob).
    A_min : int
        Minimum number of active Control Cells.
    voxel_size : float
        Base voxel size (scene scale).
    fork : int
        Octree branching factor (2 for binary octree).
    levels : int
        Number of LOD levels.
    init_pos : Tensor[3]
        Scene lower-bound corner (box_min).
    """

    def __init__(self, B_total, floor, rho_min, A_min,
                 voxel_size, fork, levels, init_pos):
        self._B_total = B_total
        self._floor = floor
        self._rho_min = rho_min
        self._A_min = A_min
        self._voxel_size = voxel_size
        self._fork = fork
        self._levels = levels
        self._init_pos = init_pos

        self._control_level = None
        self._cell_size = None

    def _ensure_control_level(self):
        if self._control_level is None:
            raise RuntimeError(
                "control_level not set; call set_control_level() before "
                "cell_id() or reduce()"
            )

    @staticmethod
    def _flatten_coords(coords):
        return (
            coords[:, 0].to(torch.int64) * _HASH_P1 +
            coords[:, 1].to(torch.int64) * _HASH_P2 +
            coords[:, 2].to(torch.int64) * _HASH_P3
        )

    def set_control_level(self, anchor_positions):
        if self._control_level is not None:
            raise RuntimeError("control_level already set")

        positions = anchor_positions.to(self._init_pos.device, copy=False)

        feasible = False
        best_level_feasible = None

        for level in range(self._levels):
            cell_size_lvl = self._voxel_size / (float(self._fork) ** level)
            cell_coords = torch.round(
                (positions - self._init_pos) / cell_size_lvl
            ).long()
            unique_cells = torch.unique(cell_coords, dim=0)
            N_active = unique_cells.shape[0]

            if N_active == 0:
                continue

            mean_occ = self._B_total / N_active

            if mean_occ >= self._rho_min and N_active >= self._A_min:
                feasible = True
                best_level_feasible = level

        if not feasible:
            best_level_feasible = None
            for level in range(self._levels):
                cell_size_lvl = self._voxel_size / (float(self._fork) ** level)
                cell_coords = torch.round(
                    (positions - self._init_pos) / cell_size_lvl
                ).long()
                unique_cells = torch.unique(cell_coords, dim=0)
                N_active = unique_cells.shape[0]
                if N_active >= self._A_min:
                    # rho_min unsatisfiable here; degrade to the COARSEST
                    # level meeting A_min (fewer cells → larger mean_occ →
                    # least rho_min violation). Do NOT pick finer: finer =
                    # emptier cells = worse. break = take coarsest match.
                    best_level_feasible = level
                    break

        if best_level_feasible is None:
            best_level_feasible = self._levels - 1
            warnings.warn(
                f"set_control_level: no level satisfies A_min={self._A_min}; "
                f"falling back to finest level {best_level_feasible}"
            )

        cell_size_chosen = self._voxel_size / (float(self._fork) ** best_level_feasible)
        cell_coords_chosen = torch.round(
            (positions - self._init_pos) / cell_size_chosen
        ).long()
        N_active_chosen = torch.unique(cell_coords_chosen, dim=0).shape[0]
        if N_active_chosen > 0 and self._floor * N_active_chosen > self._B_total:
            raise ValueError(
                f"B_total={self._B_total} too small for scene: "
                f"control level {best_level_feasible} has {N_active_chosen} "
                f"occupied cells; need B_total >= floor*N_active="
                f"{self._floor * N_active_chosen}"
            )

        self._control_level = best_level_feasible
        self._cell_size = self._voxel_size / (float(self._fork) ** self._control_level)
        return self._control_level

    def cell_id(self, anchor_positions):
        self._ensure_control_level()

        device = anchor_positions.device
        init_pos = self._init_pos.to(device)

        grid_coords = torch.round(
            (anchor_positions - init_pos) / self._cell_size
        ).long()

        return self._flatten_coords(grid_coords)

    def reduce(self, anchor_positions, weights, exclude=None):
        self._ensure_control_level()

        if weights.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=weights.device),
                torch.empty(0, dtype=weights.dtype, device=weights.device),
            )

        if exclude is not None:
            include = ~exclude
            anchor_positions = anchor_positions[include]
            weights = weights[include]

        if weights.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=weights.device),
                torch.empty(0, dtype=weights.dtype, device=weights.device),
            )

        init_pos = self._init_pos.to(anchor_positions.device)
        grid_coords = torch.round(
            (anchor_positions - init_pos) / self._cell_size
        ).long()

        unique_coords, inverse_indices = torch.unique(grid_coords, dim=0,
                                                      return_inverse=True)

        cell_ids = self._flatten_coords(unique_coords)

        if torch.unique(cell_ids).numel() != unique_coords.shape[0]:
            raise RuntimeError(
                "Spatial hash collision detected in reduce(); "
                "cell count mismatch."
            )

        d_v = torch.zeros(unique_coords.shape[0], dtype=weights.dtype,
                          device=weights.device)
        d_v.scatter_add_(0, inverse_indices, weights)

        return cell_ids, d_v


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
