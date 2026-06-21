import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types
import torch
import pytest
from demand.source_b import evaluate_source_b
from demand import PhotometricDemand
from partition import StubPartition


def _fake_cfg(period=3, camlist=2):
    return types.SimpleNamespace(b_refresh_period=period, b_camlist_size=camlist)


def test_hold_step_returns_cache_without_rendering():
    cached = {2: torch.tensor([1.0])}
    model = types.SimpleNamespace(_b_step=1, _b_cache=cached)
    calls = []
    def fake_render(*a, **k):
        calls.append(1)
        raise AssertionError("render must not be called on a hold step")
    demand_b = object()
    d_b, ms = evaluate_source_b(
        model, train_cameras=["camA", "camB", "camC"], pipe=None, bg=None,
        demand_b=demand_b, partition=None, cfg=_fake_cfg(),
        render_fn=fake_render)
    assert calls == []
    assert d_b is cached
    assert ms is None


def test_refresh_renders_zeroes_and_writes_cache():
    N = 1
    model = types.SimpleNamespace(
        _b_step=3,
        _b_cache=None,
        photometric_error_accum=torch.tensor([999.0]),
        get_anchor=torch.zeros(N, 3),
        n_offsets=1,
    )
    def fake_render(cam, m, pipe, bg):
        return {"render": torch.ones(3, 4, 4),
                "xyz": torch.tensor([[0.0, 0.0, 1.0]]),
                "radii": torch.tensor([2], dtype=torch.int32),
                "neural_opacity": torch.tensor([[1.0]]),
                "selection_mask": torch.tensor([True])}
    calls = {"n": 0}
    def counting_render(*a, **k):
        calls["n"] += 1
        return fake_render(*a, **k)
    cam = types.SimpleNamespace(original_image=torch.zeros(3, 4, 4),
                                full_proj_transform=torch.eye(4))
    d_b, ms = evaluate_source_b(
        model, [cam, cam], pipe=None, bg=None,
        demand_b=PhotometricDemand(), partition=StubPartition(),
        cfg=_fake_cfg(period=3, camlist=2),
        render_fn=counting_render)

    assert calls["n"] == 2
    assert torch.allclose(model.photometric_error_accum, torch.tensor([4.0]))
    assert set(d_b.keys()) == {0}
    assert d_b[0] == pytest.approx(4.0)
    assert d_b is model._b_cache
    assert ms >= 0.0


def test_hold_returns_identical_cache_after_refresh():
    N = 1
    model = types.SimpleNamespace(
        _b_step=3, _b_cache=None,
        photometric_error_accum=torch.zeros(N),
        get_anchor=torch.zeros(N, 3), n_offsets=1,
    )
    def refresh_render(cam, m, pipe, bg):
        return {"render": torch.ones(3, 4, 4),
                "xyz": torch.tensor([[0.0, 0.0, 1.0]]),
                "radii": torch.tensor([2], dtype=torch.int32),
                "neural_opacity": torch.tensor([[1.0]]),
                "selection_mask": torch.tensor([True])}
    cam = types.SimpleNamespace(original_image=torch.zeros(3, 4, 4),
                                full_proj_transform=torch.eye(4))
    cfg = _fake_cfg(period=3, camlist=2)
    demand_b, partition = PhotometricDemand(), StubPartition()

    d_b_refresh, _ = evaluate_source_b(model, [cam, cam], None, None,
                                       demand_b, partition, cfg,
                                       render_fn=refresh_render)

    model._b_step = 4
    def must_not_render(*a, **k):
        raise AssertionError("render called on hold step")
    d_b_hold, ms_hold = evaluate_source_b(model, [cam, cam], None, None,
                                          demand_b, partition, cfg,
                                          render_fn=must_not_render)

    assert d_b_hold is d_b_refresh
    assert d_b_hold == {0: pytest.approx(4.0)}
    assert ms_hold is None
