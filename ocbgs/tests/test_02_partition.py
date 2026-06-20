import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from partition import OctreePartition, Partition


_CONF = dict(
    B_total=1000,
    floor=1,
    rho_min=8,
    A_min=10,
    voxel_size=2.0,
    fork=2,
    levels=5,
    init_pos=torch.tensor([0.0, 0.0, 0.0]),
)


class TestOctreePartitionInstantiation:

    def test_stores_config(self):
        p = OctreePartition(**_CONF)
        assert p._B_total == _CONF['B_total']
        assert p._floor == _CONF['floor']
        assert p._rho_min == _CONF['rho_min']
        assert p._A_min == _CONF['A_min']
        assert p._voxel_size == _CONF['voxel_size']
        assert p._fork == _CONF['fork']
        assert p._levels == _CONF['levels']
        assert torch.equal(p._init_pos, _CONF['init_pos'])

    def test_control_level_not_set_initially(self):
        p = OctreePartition(**_CONF)
        assert p._control_level is None
        assert p._cell_size is None

    def test_is_partition(self):
        p = OctreePartition(**_CONF)
        assert isinstance(p, Partition)


class TestSetControlLevel:

    def test_derives_level_and_freezes(self):
        p = OctreePartition(**_CONF)

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        level = p.set_control_level(positions)
        assert isinstance(level, int)
        assert 0 <= level < _CONF['levels']
        expected_cell_size = _CONF['voxel_size'] / (_CONF['fork'] ** level)
        assert p._cell_size == expected_cell_size

    def test_called_twice_raises(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))
        with pytest.raises(RuntimeError, match="already set"):
            p.set_control_level(torch.randn(10, 3))

    def test_control_level_in_range(self):
        p = OctreePartition(**_CONF)
        level = p.set_control_level(torch.randn(50, 3) * 5.0)
        assert 0 <= level < _CONF['levels']
        assert p._cell_size == _CONF['voxel_size'] / (_CONF['fork'] ** level)

    def test_rho_min_and_A_min_satisfied(self):
        p = OctreePartition(**{**_CONF, 'rho_min': 2, 'A_min': 2})
        positions = torch.randn(100, 3)

        level = p.set_control_level(positions)
        cell_size = _CONF['voxel_size'] / (_CONF['fork'] ** level)
        cell_coords = torch.round((positions - _CONF['init_pos']) / cell_size).long()
        unique_cells = torch.unique(cell_coords, dim=0)
        N_active = unique_cells.shape[0]

        assert N_active >= p._A_min
        assert p._B_total / N_active >= p._rho_min

    def test_safety_property_floor_times_N_active_lt_B_total(self):
        p = OctreePartition(**_CONF)
        positions = torch.randn(500, 3)

        level = p.set_control_level(positions)
        cell_size = p._cell_size
        cell_coords = torch.round((positions - _CONF['init_pos']) / cell_size).long()
        N_active = torch.unique(cell_coords, dim=0).shape[0]

        assert p._floor * N_active < p._B_total

    def test_fallback_when_no_level_satisfies_rho_min(self):
        p = OctreePartition(**{**_CONF, 'B_total': 10, 'rho_min': 100, 'A_min': 1})
        positions = torch.randn(50, 3, dtype=torch.float32)
        level = p.set_control_level(positions)
        assert isinstance(level, int)
        assert 0 <= level < _CONF['levels']
        assert level == 0  # coarsest level is the least-wrong fallback
        assert p._cell_size is not None

    def test_fallback_when_A_min_unreachable(self):
        p = OctreePartition(**{**_CONF, 'A_min': 1000})
        positions = torch.randn(10, 3, dtype=torch.float32)
        with pytest.warns(UserWarning):
            level = p.set_control_level(positions)
        assert isinstance(level, int)
        assert level == _CONF['levels'] - 1
        assert p._cell_size is not None


class TestCellId:

    def test_requires_control_level(self):
        p = OctreePartition(**_CONF)
        with pytest.raises(RuntimeError, match="control_level"):
            p.cell_id(torch.randn(5, 3))

    def test_shape_and_dtype(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))
        cid = p.cell_id(torch.randn(7, 3))
        assert cid.shape == (7,)
        assert cid.dtype == torch.int64

    def test_round_semantics_not_floor(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(20, 3) * 2.0)
        cs = p._cell_size

        pos = torch.zeros(4, 3)
        pos[0, 0] = 0.3 * cs
        pos[1, 0] = 0.7 * cs
        pos[2, 0] = 1.2 * cs
        pos[3, 0] = -0.4 * cs

        cid = p.cell_id(pos)

        assert cid[0].item() == cid[3].item()
        assert cid[1].item() == cid[2].item()
        assert cid[0].item() != cid[1].item()

    def test_same_coord_gives_same_id(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        pos = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
        cid1 = p.cell_id(pos)
        cid2 = p.cell_id(pos)
        assert cid1.item() == cid2.item()

    def test_deterministic_across_batches(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        cid_batch = p.cell_id(pos)
        cid_single = torch.cat([p.cell_id(pos[i:i+1]) for i in range(pos.shape[0])])
        assert torch.equal(cid_batch, cid_single)

    def test_no_collision_on_large_grid(self):
        p = OctreePartition(**{**_CONF, 'voxel_size': 1.0, 'fork': 1,
                                'init_pos': torch.tensor([0.0, 0.0, 0.0])})
        p.set_control_level(torch.randn(10, 3))

        n = 100
        coords = torch.arange(n, dtype=torch.long)
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1).float()
        positions = grid.view(-1, 3)

        cid = p.cell_id(positions)
        unique_cid = torch.unique(cid)
        assert unique_cid.numel() == n ** 3


class TestReduce:

    def test_requires_control_level(self):
        p = OctreePartition(**_CONF)
        with pytest.raises(RuntimeError, match="control_level"):
            p.reduce(torch.randn(5, 3), torch.ones(5))

    def test_unit_weights_gives_occupancy(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        weights = torch.ones(5)

        cell_ids, d_v = p.reduce(positions, weights)

        assert cell_ids.numel() <= 2
        assert d_v.shape == cell_ids.shape
        assert d_v.sum().item() == pytest.approx(5.0)

    def test_weighted_values_gives_correct_segment_sums(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ], dtype=torch.float32)
        weights = torch.tensor([1.0, 2.0, 5.0])

        cell_ids, d_v = p.reduce(positions, weights)

        assert d_v.sum().item() == pytest.approx(8.0)

        unique_cells = torch.unique(cell_ids)
        assert unique_cells.numel() == 2

    def test_exclude_mask(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ], dtype=torch.float32)
        weights = torch.tensor([1.0, 2.0, 3.0, 10.0])
        exclude = torch.tensor([False, True, False, False])

        cell_ids, d_v = p.reduce(positions, weights, exclude=exclude)

        assert d_v.sum().item() == pytest.approx(14.0)

        cell_ids_all, d_v_all = p.reduce(positions, weights)
        assert d_v_all.sum().item() == pytest.approx(16.0)
        assert d_v.sum().item() < d_v_all.sum().item()

    def test_exclude_none_anchors(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        positions = torch.tensor([
            [0.0, 0.0, 0.0],
        ], dtype=torch.float32)
        weights = torch.tensor([5.0])
        exclude = torch.tensor([True])

        cell_ids, d_v = p.reduce(positions, weights, exclude=exclude)
        assert d_v.numel() == 0
        assert cell_ids.numel() == 0

    def test_empty_weights(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        cell_ids, d_v = p.reduce(torch.empty(0, 3), torch.empty(0))
        assert cell_ids.shape == (0,)
        assert d_v.shape == (0,)
        assert cell_ids.dtype == torch.int64


class TestCellIdReduceConsistency:

    def test_cell_id_and_reduce_align(self):
        p = OctreePartition(**_CONF)
        p.set_control_level(torch.randn(10, 3))

        N = 50
        positions = torch.rand(N, 3) * 10.0
        weights = torch.rand(N)

        cell_ids, d_v = p.reduce(positions, weights)

        cid_map = p.cell_id(positions)
        unique_from_cid = torch.unique(cid_map)
        assert unique_from_cid.numel() == cell_ids.numel()
        assert torch.equal(torch.sort(unique_from_cid).values,
                           torch.sort(cell_ids).values)
