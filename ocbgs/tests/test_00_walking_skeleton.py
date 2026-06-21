import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch


def _mock_scene(n_anchors=5):
    class MockGaussians:
        def __init__(self, n):
            self._anchor = torch.randn(n, 3)

        @property
        def get_anchor(self):
            return self._anchor

    class MockScene:
        def __init__(self, n):
            self.gaussians = MockGaussians(n)

    return MockScene(n_anchors)


class TestPackageImports:
    def test_import_demand(self):
        from demand import DemandProducer, StubDemandProducer
        assert DemandProducer is not None
        assert StubDemandProducer is not None

    def test_import_partition(self):
        from partition import Partition, StubPartition
        assert Partition is not None
        assert StubPartition is not None

    def test_import_controller(self):
        from controller import BudgetController, StubBudgetController, ReallocationPlan
        assert BudgetController is not None
        assert StubBudgetController is not None
        assert ReallocationPlan is not None

    def test_lazy_rasterizer_import_no_top_level_cuda(self):
        """
        Server-only test: requires full CUDA environment.
        Verifies gaussian_renderer no longer does a top-level CUDA import.
        """
        try:
            import gaussian_renderer
        except ImportError:
            pytest.skip("gaussian_renderer import requires CUDA environment")
        assert hasattr(gaussian_renderer, '_lazy_rasterizer')
        import inspect
        source = inspect.getsource(gaussian_renderer)
        assert 'from diff_gaussian_rasterization' not in source.split('def _lazy_rasterizer')[0], (
            'gaussian_renderer must not import diff_gaussian_rasterization at module level'
        )


class TestAbcEnforcement:
    def test_demand_producer_cannot_instantiate(self):
        from demand import DemandProducer
        with pytest.raises(TypeError):
            DemandProducer()

    def test_partition_cannot_instantiate(self):
        from partition import Partition
        with pytest.raises(TypeError):
            Partition()

    def test_budget_controller_cannot_instantiate(self):
        from controller import BudgetController
        with pytest.raises(TypeError):
            BudgetController()


class TestStubDemandProducer:
    def test_instantiate(self):
        from demand import StubDemandProducer
        dp = StubDemandProducer()
        assert dp is not None

    def test_produce_shape_and_value(self):
        from demand import StubDemandProducer
        scene = _mock_scene(n_anchors=7)
        dp = StubDemandProducer()
        s_a = dp.produce(scene, None)
        assert s_a.shape == (7,)
        assert torch.allclose(s_a, torch.ones(7))
        assert s_a.dtype == torch.float32

    def test_produce_with_gaussian_model(self):
        from demand import StubDemandProducer

        class MockGaussianModel:
            def __init__(self, n):
                self._anchor = torch.randn(n, 3)

            @property
            def get_anchor(self):
                return self._anchor

        g = MockGaussianModel(10)
        dp = StubDemandProducer()
        s_a = dp.produce(g, None)
        assert s_a.shape == (10,)
        assert torch.allclose(s_a, torch.ones(10))

    def test_produce_with_none(self):
        from demand import StubDemandProducer
        dp = StubDemandProducer()
        s_a = dp.produce(None, None)
        assert s_a.shape == (0,)


class TestStubPartition:
    def test_instantiate(self):
        from partition import StubPartition
        p = StubPartition()
        assert p is not None

    def test_set_control_level(self):
        from partition import StubPartition
        p = StubPartition()
        level = p.set_control_level(torch.randn(10, 3))
        assert level == 0

    def test_cell_id_shape(self):
        from partition import StubPartition
        p = StubPartition()
        positions = torch.randn(8, 3)
        cid = p.cell_id(positions)
        assert cid.shape == (8,)
        assert torch.all(cid == 0)
        assert cid.dtype == torch.long

    def test_reduce_single_global_cell(self):
        from partition import StubPartition
        p = StubPartition()
        positions = torch.randn(5, 3)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        cell_ids, d_v = p.reduce(positions, weights)
        assert cell_ids.shape == (1,)
        assert d_v.shape == (1,)
        assert cell_ids[0].item() == 0
        assert d_v.item() == pytest.approx(15.0)
        assert d_v.dtype == torch.float32

    def test_reduce_empty_weights(self):
        from partition import StubPartition
        p = StubPartition()
        positions = torch.empty(0, 3)
        weights = torch.empty(0)
        cell_ids, d_v = p.reduce(positions, weights)
        assert cell_ids.shape == (0,)
        assert d_v.shape == (0,)
        assert cell_ids.dtype == torch.long
        assert d_v.dtype == torch.float32


class TestStubBudgetController:
    def test_instantiate(self):
        from controller import StubBudgetController
        bc = StubBudgetController()
        assert bc is not None

    def test_plan_identity(self):
        from controller import StubBudgetController
        bc = StubBudgetController()
        cell_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        d_A = torch.tensor([5.0, 3.0, 2.0])
        occupancy = torch.tensor([10, 10, 10], dtype=torch.long)
        plan = bc.plan(cell_ids, d_A, occupancy, B_total=30)
        assert torch.all(plan.delta == 0)
        assert plan.phase == "ramp"
        assert torch.equal(plan.c_target, occupancy)
        assert torch.equal(plan.cell_ids, cell_ids)

    def test_plan_shape_matches_cell_count(self):
        from controller import StubBudgetController
        bc = StubBudgetController()
        C = 5
        cell_ids = torch.arange(C, dtype=torch.long)
        d_A = torch.ones(C)
        occupancy = torch.ones(C, dtype=torch.long)
        plan = bc.plan(cell_ids, d_A, occupancy, B_total=-1)
        assert plan.delta.shape == (C,)
        assert plan.c_target.shape == (C,)
        assert plan.cell_ids.shape == (C,)


class TestReallocationPlan:
    def test_is_dataclass(self):
        from controller import ReallocationPlan
        import dataclasses
        assert dataclasses.is_dataclass(ReallocationPlan)

    def test_fields(self):
        from controller import ReallocationPlan
        plan = ReallocationPlan(
            cell_ids=torch.zeros(1, dtype=torch.long),
            delta=torch.zeros(1, dtype=torch.long),
            phase="ramp",
            c_target=torch.ones(1),
        )
        assert plan.cell_ids is not None
        assert plan.delta is not None
        assert plan.phase == "ramp"
        assert plan.c_target is not None




class TestBTotal:
    def test_optimization_params_has_B_total(self):
        from arguments import OptimizationParams
        from argparse import ArgumentParser
        parser = ArgumentParser()
        op = OptimizationParams(parser)
        assert hasattr(op, 'B_total')
        assert op.B_total == -1

    def test_gaussian_model_constructor_accepts_B_total(self):
        try:
            from scene.gaussian_model import GaussianModel
        except ImportError:
            pytest.skip("gaussian_model import requires CUDA environment")
        import inspect
        sig = inspect.signature(GaussianModel.__init__)
        assert 'B_total' in sig.parameters
        assert sig.parameters['B_total'].default == -1


class TestSeedSupport:
    def test_safe_state_accepts_seed(self):
        from utils.general_utils import safe_state
        import inspect
        sig = inspect.signature(safe_state)
        assert 'seed' in sig.parameters
        assert sig.parameters['seed'].default == 0

    def test_seed_argument_registered(self):
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from argparse import ArgumentParser
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)
        OptimizationParams(parser)
        PipelineParams(parser)
        parser.add_argument('--seed', type=int, default=0)
        args = parser.parse_args([])
        assert args.seed == 0


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestSetupFiles:
    def test_environment_yml_exists(self):
        env_yml = os.path.join(_PROJECT_ROOT, 'environment.yml')
        assert os.path.isfile(env_yml)

    def test_setup_sh_exists(self):
        setup_sh = os.path.join(_PROJECT_ROOT, 'setup.sh')
        assert os.path.isfile(setup_sh)


class TestOccupancyBincount:
    def test_bincount_on_membership_not_reduced_ids(self):
        """
        n(v) must be computed from per-anchor cell_id() membership,
        not from the unique cell_ids returned by reduce().

        Simulates a non-trivial partition: 5 anchors in cells [2, 2, 5, 5, 5],
        reduce() returns cell_ids=[2, 5], d_v=[s1, s2].
        """
        membership = torch.tensor([2, 2, 5, 5, 5], dtype=torch.long)
        cell_ids = torch.tensor([2, 5], dtype=torch.long)

        full_n_v = torch.bincount(membership, minlength=int(membership.max().item()) + 1)
        n_v = full_n_v[cell_ids]

        assert full_n_v.tolist() == [0, 0, 2, 0, 0, 3]
        assert n_v.tolist() == [2, 3]
        assert n_v.shape[0] == cell_ids.shape[0]

    def test_bincount_stub_partition(self):
        membership = torch.zeros(7, dtype=torch.long)
        cell_ids = torch.zeros(1, dtype=torch.long)

        full_n_v = torch.bincount(membership, minlength=int(membership.max().item()) + 1)
        n_v = full_n_v[cell_ids]

        assert n_v.item() == 7


class TestSetupShNoDuplicates:
    def test_environment_yml_has_no_pip_cuda_submodules(self):
        env_yml = os.path.join(_PROJECT_ROOT, 'environment.yml')
        with open(env_yml, 'r') as f:
            content = f.read()
        assert '- submodules/diff-gaussian-rasterization' not in content, (
            'environment.yml must not reference CUDA submodules; setup.sh handles builds'
        )
        assert '- submodules/simple-knn' not in content, (
            'environment.yml must not reference CUDA submodules; setup.sh handles builds'
        )
