#
# Near/far test-view split evaluation (no re-training, no re-rendering).
#
# Reuses the per-view PSNR/SSIM/LPIPS already written to <model_path>/per_view.json
# by metrics.evaluate(), and splits the test views by camera distance so we can ask:
# does A+B beat A-only specifically on the FAR / detail views (Source B's thesis,
# 2026-06-19 spec §4.1), even when the overall average is a tie?
#
# Distance proxy is model-independent (camera centre -> centroid of test cameras),
# so the same split applies identically to every arm. For BungeeNeRF this tracks
# the satellite(far)->ground(near) scale axis.
#
# Usage (server, GPU only to load the scene/ply; no rendering):
#   python ocbgs/eval_view_split.py -m /root/autodl-tmp/relax/amsterdam/b_relax01
#   python ocbgs/eval_view_split.py -m /root/autodl-tmp/relax/amsterdam/c_relax01
# Optional: --far_frac 0.33 to compare farthest third vs nearest third.
#
import os
import json
import numpy as np
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


def split_camera_centers(dataset, iteration, split):
    """Load the cameras of the requested split in the SAME order train.py/render.py
    saved the PNGs, and return their world-space centres as an [N,3] numpy array.

    train.py renders + evaluates the TRAIN split when eval=False and the TEST split
    when eval=True (train.py:594-603). per_view.json is keyed 00000.png.. in that
    split's camera order, so split here must match how the run was evaluated."""
    dataset.eval = (split == "test")
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,
    )
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,
                  resolution_scales=dataset.resolution_scales)
    cams = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    centers = np.stack([c.camera_center.detach().cpu().numpy() for c in cams], axis=0)
    return centers, scene.loaded_iter


def load_per_view_metric(model_path, metric="PSNR"):
    """Return {filename: value} for the requested metric from per_view.json."""
    path = os.path.join(model_path, "per_view.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the standard eval first so per_view.json exists:\n"
            f"  python ocbgs/metrics.py -m {model_path}")
    with open(path) as fp:
        pv = json.load(fp)
    method = next(iter(pv))  # e.g. "ours_30000"
    return pv[method][metric], method


def group_mean(value_map, indices):
    vals = [value_map["{:05d}.png".format(i)] for i in indices]
    return float(np.mean(vals)), len(vals)


def main():
    parser = ArgumentParser(description="Near/far test-view split evaluation")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--far_frac", default=0.5, type=float,
                        help="fraction of farthest views in the FAR group "
                             "(0.5 = median split; 0.33 = farthest third vs nearest third)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="which split per_view.json holds — runs with --eval evaluate "
                             "'test', runs without evaluate 'train' (train.py:594-603)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    model_path = dataset.model_path

    centers, loaded_iter = split_camera_centers(dataset, args.iteration, args.split)

    # Hard guard: the camera list MUST match per_view.json one-to-one, or the
    # index->camera->distance pairing is garbage (this silently produced wrong
    # numbers before the guard existed). The 'all' column reproducing the headline
    # PSNR is the second, independent check.
    psnr_map, _ = load_per_view_metric(model_path, "PSNR")
    if len(centers) != len(psnr_map):
        raise RuntimeError(
            f"camera/per_view mismatch: --split {args.split} gives {len(centers)} cameras "
            f"but per_view.json has {len(psnr_map)} views. The run was evaluated on the OTHER "
            f"split — re-run with --split {'test' if args.split=='train' else 'train'}, or "
            f"regenerate per_view.json for this split.")
    scene_center = centers.mean(axis=0)
    dists = np.linalg.norm(centers - scene_center, axis=1)
    order = np.argsort(dists)  # ascending: near -> far
    N = len(dists)

    k = max(1, int(round(args.far_frac * N)))
    near_idx = order[:k].tolist()          # nearest k
    far_idx = order[-k:].tolist()          # farthest k
    all_idx = list(range(N))

    print(f"\n=== view-split eval: {model_path} (iter {loaded_iter}) ===")
    print(f"test views: {N} | dist min/med/max = "
          f"{dists.min():.3f}/{np.median(dists):.3f}/{dists.max():.3f} | far_frac={args.far_frac}")
    print(f"near group: {k} views (dist<= {dists[order[k-1]]:.3f}) | "
          f"far group: {k} views (dist>= {dists[order[-k]]:.3f})\n")

    out = {"model_path": model_path, "iteration": loaded_iter, "n_views": N,
           "far_frac": args.far_frac}
    header = f"{'metric':<7}{'all':>12}{'near':>12}{'far':>12}{'far-near':>12}"
    print(header)
    print("-" * len(header))
    for metric in ("PSNR", "SSIM", "LPIPS"):
        vmap, _ = load_per_view_metric(model_path, metric)
        all_v, _ = group_mean(vmap, all_idx)
        near_v, _ = group_mean(vmap, near_idx)
        far_v, _ = group_mean(vmap, far_idx)
        print(f"{metric:<7}{all_v:>12.5f}{near_v:>12.5f}{far_v:>12.5f}{far_v - near_v:>+12.5f}")
        out[metric] = {"all": all_v, "near": near_v, "far": far_v}

    out_path = os.path.join(model_path, "view_split_eval.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
