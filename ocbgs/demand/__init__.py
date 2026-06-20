from abc import ABC, abstractmethod
import torch


class DemandProducer(ABC):

    @abstractmethod
    def produce(self, scene, stats):
        """
        Produce per-anchor Anchor Demand s(a).

        Args:
            scene: the Scene object (for camera info, not for anchors)
            stats: training statistics dict (or None for degenerate stub)

        Returns:
            s_a: Tensor[N] of per-anchor demand scores, non-negative
        """
        ...


class StubDemandProducer(DemandProducer):

    def produce(self, scene, stats):
        if scene is not None and hasattr(scene, 'gaussians'):
            anchor = scene.gaussians.get_anchor
        elif scene is not None and hasattr(scene, 'get_anchor'):
            anchor = scene.get_anchor
        else:
            anchor = torch.empty(0)
        N = anchor.shape[0]
        return torch.ones(N, device=anchor.device) if N > 0 else anchor.new_zeros(0)
