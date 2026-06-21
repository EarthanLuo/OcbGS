import random
import time
import torch
from demand import PhotometricDemand, KEY_PHOTOMETRIC_ERROR_ACCUM


def evaluate_source_b(model, train_cameras, pipe, bg, demand_b, partition, cfg, render_fn=None):
    if model._b_step % cfg.b_refresh_period != 0:
        return model._b_cache, None
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    camlist = random.sample(train_cameras, min(cfg.b_camlist_size, len(train_cameras)))
    t0 = time.perf_counter()
    N = model.get_anchor.shape[0]
    model.photometric_error_accum = torch.zeros(N, device=model.get_anchor.device)
    with torch.no_grad():
        for cam in camlist:
            pkg = render_fn(cam, model, pipe, bg)
            err_map = (pkg["render"] - cam.original_image.to(pkg["render"].device)).abs().mean(0)
            err_delta, _ = PhotometricDemand.accumulate_view(
                err_map, pkg["xyz"], pkg["radii"], pkg["neural_opacity"],
                pkg["selection_mask"], cam.full_proj_transform,
                model.n_offsets, torch.arange(N), N)
            model.photometric_error_accum += err_delta
    render_ms = (time.perf_counter() - t0) * 1000.0
    s_b = demand_b.produce(scene=None, stats={KEY_PHOTOMETRIC_ERROR_ACCUM: model.photometric_error_accum})
    cell_ids, d_v = partition.reduce(model.get_anchor, s_b)
    model._b_cache = {int(c): float(v) for c, v in zip(cell_ids.tolist(), d_v.tolist())}
    print(f"[SOURCE_B] refresh | s_B.shape={s_b.shape} s_B.min()={s_b.min().item():.4f} "
          f"d_B_cells={len(model._b_cache)} render_ms={render_ms:.1f}")
    return model._b_cache, render_ms
