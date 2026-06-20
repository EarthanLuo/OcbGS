import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from demand import ErrorVisibilityDemand, KEY_ANCHOR_DEMON, KEY_OFFSET_GRADIENT_ACCUM, KEY_OFFSET_DENOM


def _make_stats(n_anchors, n_offsets,
                anchor_demon_vals=None,
                offset_grad_vals=None,
                offset_denom_vals=None,
                device=None):
    if device is None:
        device = torch.device('cpu')
    stats = {}
    if anchor_demon_vals is not None:
        stats[KEY_ANCHOR_DEMON] = torch.tensor(anchor_demon_vals, dtype=torch.float32, device=device).view(-1, 1)
    else:
        stats[KEY_ANCHOR_DEMON] = torch.ones(n_anchors, 1, dtype=torch.float32, device=device)
    k = n_anchors * n_offsets
    if offset_grad_vals is not None:
        stats[KEY_OFFSET_GRADIENT_ACCUM] = torch.tensor(offset_grad_vals, dtype=torch.float32, device=device).view(-1, 1)
    else:
        stats[KEY_OFFSET_GRADIENT_ACCUM] = torch.zeros(k, 1, dtype=torch.float32, device=device)
    if offset_denom_vals is not None:
        stats[KEY_OFFSET_DENOM] = torch.tensor(offset_denom_vals, dtype=torch.float32, device=device).view(-1, 1)
    else:
        stats[KEY_OFFSET_DENOM] = torch.zeros(k, 1, dtype=torch.float32, device=device)
    return stats


def _mock_scene():
    class MockScene:
        pass
    return MockScene()


class TestErrorVisibilityDemandInstantiation:

    def test_instantiate_with_defaults(self):
        dp = ErrorVisibilityDemand()
        assert dp.check_interval == 100
        assert dp.success_threshold == 0.8

    def test_instantiate_with_custom_params(self):
        dp = ErrorVisibilityDemand(check_interval=50, success_threshold=0.6)
        assert dp.check_interval == 50
        assert dp.success_threshold == 0.6

    def test_is_demand_producer(self):
        from demand import DemandProducer
        dp = ErrorVisibilityDemand()
        assert isinstance(dp, DemandProducer)


class TestErrorVisibilityDemandProduce:

    def test_shape_and_nonnegative(self):
        scene = _mock_scene()
        n, k = 5, 10
        stats = _make_stats(n, k)
        dp = ErrorVisibilityDemand()
        s_a = dp.produce(scene, stats)
        assert s_a.shape == (n,)
        assert (s_a >= 0).all()
        assert s_a.dtype == torch.float32

    def test_masked_max_single_mature_offset(self):
        scene = _mock_scene()
        n, k = 3, 2
        stats = _make_stats(n, k,
                            anchor_demon_vals=[1.0, 1.0, 1.0],
                            offset_grad_vals=[
                                50.0, 99.0,
                                99.0, 0.0,
                                0.0, 0.0,
                            ],
                            offset_denom_vals=[
                                100.0, 1.0,
                                1.0, 100.0,
                                0.0, 0.0,
                            ])
        dp = ErrorVisibilityDemand(check_interval=10, success_threshold=0.8)
        s_a = dp.produce(scene, stats)

        assert s_a[0].item() == pytest.approx(0.5)
        assert s_a[1].item() == pytest.approx(0.0)
        assert s_a[2].item() == pytest.approx(0.0)

    def test_all_immature_offsets_yield_zero(self):
        scene = _mock_scene()
        n, k = 2, 3
        stats = _make_stats(n, k,
                            anchor_demon_vals=[5.0, 3.0],
                            offset_grad_vals=[10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
                            offset_denom_vals=[1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
                            )
        dp = ErrorVisibilityDemand(check_interval=100, success_threshold=0.8)
        s_a = dp.produce(scene, stats)
        assert s_a[0].item() == pytest.approx(0.0)
        assert s_a[1].item() == pytest.approx(0.0)

    def test_zero_anchor_demon_yields_zero(self):
        scene = _mock_scene()
        n, k = 3, 2
        stats = _make_stats(n, k,
                            anchor_demon_vals=[0.0, 0.0, 5.0],
                            offset_grad_vals=[50.0, 0.0, 30.0, 0.0, 80.0, 0.0],
                            offset_denom_vals=[100.0, 0.0, 100.0, 0.0, 100.0, 0.0],
                            )
        dp = ErrorVisibilityDemand(check_interval=10, success_threshold=0.8)
        s_a = dp.produce(scene, stats)
        assert s_a[0].item() == pytest.approx(0.0)
        assert s_a[1].item() == pytest.approx(0.0)
        assert s_a[2].item() == pytest.approx(4.0)

    def test_error_times_visibility_agrees_with_hand_compute(self):
        scene = _mock_scene()
        n, k = 4, 3

        anchor_demon_vals = [10.0, 5.0, 0.0, 2.0]
        offset_grad_vals = [
            80.0, 16.0, 72.0,
            24.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]
        offset_denom_vals = [
            80.0, 80.0, 80.0,
            80.0, 80.0, 80.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ]

        stats = _make_stats(n, k,
                            anchor_demon_vals=anchor_demon_vals,
                            offset_grad_vals=offset_grad_vals,
                            offset_denom_vals=offset_denom_vals)

        dp = ErrorVisibilityDemand(check_interval=10, success_threshold=0.8)

        s_a = dp.produce(scene, stats)

        assert s_a.shape == (n,)
        assert (s_a >= 0).all()

        expected_errors = [1.0, 0.3, 0.0, 0.0]
        expected_demons = anchor_demon_vals
        expected_s = [e * d for e, d in zip(expected_errors, expected_demons)]

        for i in range(n):
            assert s_a[i].item() == pytest.approx(expected_s[i]), f"anchor {i} mismatch"

    def test_empty_anchors(self):
        scene = _mock_scene()
        stats = _make_stats(0, 10)
        dp = ErrorVisibilityDemand()
        s_a = dp.produce(scene, stats)
        assert s_a.shape == (0,)
        assert s_a.numel() == 0

    def test_nan_in_gradient_due_to_zero_denom(self):
        scene = _mock_scene()
        n, k = 2, 1
        stats = _make_stats(n, k,
                            anchor_demon_vals=[1.0, 1.0],
                            offset_grad_vals=[0.0, 99.0],
                            offset_denom_vals=[0.0, 0.0],
                            )
        dp = ErrorVisibilityDemand()
        s_a = dp.produce(scene, stats)
        assert s_a[0].item() == pytest.approx(0.0)
        assert s_a[1].item() == pytest.approx(0.0)


class TestKeyConstants:

    def test_keys_are_strings(self):
        assert isinstance(KEY_ANCHOR_DEMON, str)
        assert isinstance(KEY_OFFSET_GRADIENT_ACCUM, str)
        assert isinstance(KEY_OFFSET_DENOM, str)

    def test_keys_match_native_accumulator_names(self):
        assert KEY_ANCHOR_DEMON == "anchor_demon"
        assert KEY_OFFSET_GRADIENT_ACCUM == "offset_gradient_accum"
        assert KEY_OFFSET_DENOM == "offset_denom"
