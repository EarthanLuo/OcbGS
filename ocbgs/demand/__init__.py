from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

# Mapping keys for the stats dict passed to DemandProducer.produce().
# Issue 05 (adjust_anchor integration) feeds these exact names into the
# stats dict so the producer can consume them with zero coupling to the
# GaussianModel internals.
KEY_ANCHOR_DEMON = "anchor_demon"
KEY_OFFSET_GRADIENT_ACCUM = "offset_gradient_accum"
KEY_OFFSET_DENOM = "offset_denom"
KEY_PHOTOMETRIC_ERROR_ACCUM = "photometric_error_accum"


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


class ErrorVisibilityDemand(DemandProducer):
    """Demand Producer Source A: s(a) = error(a) x visibility(a).

    error(a) = masked-max over offsets of per-offset mean gradient,
    where the maturity mask reuses the native offset_mask gate
    (offset_denom > check_interval * success_threshold * 0.5).

    visibility(a) = raw anchor_demon (per-anchor view count),
    NOT normalised to [0,1] -- preserves multi-view weight.

    Partition-agnostic: knows nothing of Control Cells, control_level,
    or the Capacity Budget.
    """

    def __init__(self, check_interval=100, success_threshold=0.8):
        self.check_interval = check_interval
        self.success_threshold = success_threshold

    def produce(self, scene, stats):
        anchor_demon = stats[KEY_ANCHOR_DEMON]  # [N, 1]
        N = anchor_demon.shape[0]
        if N == 0:
            return anchor_demon.new_zeros(0)

        offset_gradient_accum = stats[KEY_OFFSET_GRADIENT_ACCUM]  # [N*k, 1]
        offset_denom = stats[KEY_OFFSET_DENOM]                    # [N*k, 1]

        n_offsets = offset_denom.shape[0] // N
        assert n_offsets > 0, "n_offsets must be positive (derived from stats shapes)"

        grads = offset_gradient_accum / offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = grads.squeeze(-1)  # [N*k]

        mature_threshold = self.check_interval * self.success_threshold * 0.5
        offset_mask = (offset_denom.squeeze(-1) > mature_threshold)  # [N*k]

        grads_norm_2d = grads_norm.view(N, n_offsets)
        offset_mask_2d = offset_mask.view(N, n_offsets)

        grads_norm_2d[~offset_mask_2d] = 0.0
        error = grads_norm_2d.max(dim=-1).values  # [N]

        visibility = anchor_demon.squeeze(-1)  # [N]

        return error * visibility


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


class PhotometricDemand(DemandProducer):

    def __init__(self):
        pass

    def produce(self, scene, stats):
        return stats[KEY_PHOTOMETRIC_ERROR_ACCUM]

    @staticmethod
    def _project_to_ndc(xyz, proj_matrix):
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        clip = xyz_hom @ proj_matrix
        w = clip[:, 3]
        w_safe = torch.where(w > 0, w, torch.ones_like(w))
        ndc = clip[:, :2] / w_safe.unsqueeze(-1)
        return ndc, w

    @staticmethod
    def accumulate_view(error_map, xyz, radii, neural_opacity, mask,
                        proj_matrix, n_offsets, visible_anchor_global_idx,
                        N_anchors):
        device = error_map.device
        err_delta = torch.zeros(N_anchors, device=device)
        cnt_delta = torch.zeros(N_anchors, device=device)
        P = xyz.shape[0]
        if P == 0:
            return err_delta, cnt_delta

        opacity_masked = neural_opacity[mask].squeeze(-1)

        ndc, w = PhotometricDemand._project_to_ndc(xyz, proj_matrix)
        onscreen = (w > 0).float()

        H, W = error_map.shape[-2], error_map.shape[-1]
        sampled = F.grid_sample(error_map.view(1, 1, H, W),
                                ndc.view(1, P, 1, 2),
                                align_corners=False,
                                padding_mode='zeros').view(P)

        visibility = (radii > 0).float()
        contrib = sampled * radii.float() * opacity_masked * visibility * onscreen

        f_indices = mask.nonzero(as_tuple=True)[0]
        global_anchor = visible_anchor_global_idx[f_indices // n_offsets]
        err_delta.scatter_add_(0, global_anchor, contrib)
        cnt_delta.scatter_add_(0, global_anchor, visibility * onscreen)
        return err_delta, cnt_delta
