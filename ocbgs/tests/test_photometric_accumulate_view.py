import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from demand import PhotometricDemand


def test_single_gaussian_screen_center_uniform_error():
    error_map = torch.ones(4, 4)
    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    proj_matrix = torch.eye(4)
    radii = torch.tensor([3], dtype=torch.int32)
    neural_opacity = torch.tensor([[0.5]])
    mask = torch.tensor([True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0])
    N_anchors = 1

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj_matrix, n_offsets, visible_anchor_global_idx,
        N_anchors,
    )

    assert err_delta.shape == (N_anchors,)
    assert torch.allclose(err_delta, torch.tensor([1.5]))
    assert torch.allclose(cnt_delta, torch.tensor([1.0]))


def test_projection_convention_no_transpose():
    xyz = torch.tensor([[1.0, 0.0, 2.0]])
    proj = torch.tensor([
        [1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.5, 0.0, 1.0],
    ])
    ndc, w = PhotometricDemand._project_to_ndc(xyz, proj)

    assert torch.allclose(w, torch.tensor([3.0]))
    assert torch.allclose(ndc, torch.tensor([[0.5, 0.5]]))


def test_grid_sample_y_axis_matches_ndc():
    error_map = torch.empty(4, 4)
    error_map[:2, :] = 2.0
    error_map[2:, :] = 4.0
    xyz = torch.tensor([[0.0, -0.5, 1.0],
                        [0.0,  0.5, 1.0]])
    proj = torch.eye(4)
    radii = torch.tensor([1, 1], dtype=torch.int32)
    neural_opacity = torch.tensor([[1.0], [1.0]])
    mask = torch.tensor([True, True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0, 1])
    N_anchors = 2

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_delta, torch.tensor([2.0, 4.0]))
    assert torch.allclose(cnt_delta, torch.tensor([1.0, 1.0]))


def test_behind_camera_zero_contribution():
    proj = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    xyz = torch.tensor([[0.0, 0.0,  0.0],
                        [0.0, 0.0, -1.0]])
    error_map = torch.ones(4, 4)
    radii = torch.tensor([2, 2], dtype=torch.int32)
    neural_opacity = torch.tensor([[0.8], [0.9]])
    mask = torch.tensor([True, True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0, 1])
    N_anchors = 2

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_delta, torch.zeros(2))
    assert torch.allclose(cnt_delta, torch.zeros(2))


def test_two_level_index_offsets_to_same_anchor():
    mask = torch.tensor([True, False, False, True, True, False])
    n_offsets = 3
    visible_anchor_global_idx = torch.tensor([2, 5])
    N_anchors = 6

    xyz = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    proj = torch.eye(4)
    error_map = torch.ones(4, 4)
    radii = torch.tensor([1, 2, 3], dtype=torch.int32)
    neural_opacity = torch.ones(6, 1)

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_delta, torch.tensor([0., 0., 1., 0., 0., 5.]))
    assert torch.allclose(cnt_delta, torch.tensor([0., 0., 1., 0., 0., 2.]))


def test_single_anchor_all_offsets_visible():
    n_offsets = 3
    mask = torch.tensor([True, True, True])
    visible_anchor_global_idx = torch.tensor([0])
    N_anchors = 1

    xyz = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    proj = torch.eye(4)
    error_map = torch.ones(4, 4)
    radii = torch.tensor([1, 2, 3], dtype=torch.int32)
    neural_opacity = torch.ones(3, 1)

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_delta, torch.tensor([6.0]))
    assert torch.allclose(cnt_delta, torch.tensor([3.0]))


def test_empty_mask_returns_zeros():
    mask = torch.zeros(6, dtype=torch.bool)
    xyz = torch.zeros(0, 3)
    radii = torch.zeros(0, dtype=torch.int32)
    neural_opacity = torch.ones(6, 1)
    n_offsets = 3
    visible_anchor_global_idx = torch.tensor([0, 1])
    N_anchors = 2
    error_map = torch.ones(4, 4)
    proj = torch.eye(4)

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert err_delta.shape == (N_anchors,)
    assert cnt_delta.shape == (N_anchors,)
    assert torch.allclose(err_delta, torch.zeros(N_anchors))
    assert torch.allclose(cnt_delta, torch.zeros(N_anchors))


def test_all_radii_zero_contrib_zero():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]])
    radii = torch.zeros(2, dtype=torch.int32)
    neural_opacity = torch.tensor([[0.5], [0.8]])
    mask = torch.tensor([True, True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0, 1])
    N_anchors = 2
    error_map = torch.ones(4, 4)
    proj = torch.eye(4)

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_delta, torch.zeros(N_anchors))
    assert torch.allclose(cnt_delta, torch.zeros(N_anchors))


def test_nonnegative_outputs():
    error_map = torch.rand(8, 8) + 0.2
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.3, 1.0], [-0.4, 0.5, 1.0]])
    radii = torch.tensor([3, 1, 5], dtype=torch.int32)
    neural_opacity = torch.tensor([[0.3], [0.9], [0.1]])
    mask = torch.tensor([True, True, True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0, 1, 2])
    N_anchors = 3
    proj = torch.eye(4)

    err_delta, cnt_delta = PhotometricDemand.accumulate_view(
        error_map, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert err_delta.min() >= 0.0
    assert cnt_delta.min() >= 0.0


def test_double_error_map_doubles_err_delta():
    error_map_a = torch.ones(4, 4)
    error_map_b = error_map_a * 2.0
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
    radii = torch.tensor([1, 2], dtype=torch.int32)
    neural_opacity = torch.tensor([[0.5], [0.5]])
    mask = torch.tensor([True, True])
    n_offsets = 1
    visible_anchor_global_idx = torch.tensor([0, 1])
    N_anchors = 2
    proj = torch.eye(4)

    err_a, cnt_a = PhotometricDemand.accumulate_view(
        error_map_a, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)
    err_b, cnt_b = PhotometricDemand.accumulate_view(
        error_map_b, xyz, radii, neural_opacity, mask,
        proj, n_offsets, visible_anchor_global_idx, N_anchors)

    assert torch.allclose(err_b, err_a * 2.0)
    assert torch.allclose(cnt_b, cnt_a)
