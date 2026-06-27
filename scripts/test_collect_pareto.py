import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "collect_results",
    os.path.join(os.path.dirname(__file__), "collect_results.py"),
)
collect_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_results)
aggregate_pareto_points = collect_results.aggregate_pareto_points


def test_aggregates_mean_and_sorts_by_arm_then_anchors():
    raw = [
        {"arm": "uniform", "factor": 1.0, "anchors": [100, 120],
         "metrics": [{"PSNR": 28.0, "SSIM": 0.80, "LPIPS": 0.20},
                     {"PSNR": 28.4, "SSIM": 0.82, "LPIPS": 0.18}]},
        {"arm": "demand", "factor": 0.5, "anchors": [50, 50],
         "metrics": [{"PSNR": 27.0, "SSIM": 0.78, "LPIPS": 0.22},
                     {"PSNR": 27.0, "SSIM": 0.78, "LPIPS": 0.22}]},
        {"arm": "demand", "factor": 1.0, "anchors": [110, 90],
         "metrics": [{"PSNR": 28.5, "SSIM": 0.83, "LPIPS": 0.17},
                     {"PSNR": 28.5, "SSIM": 0.83, "LPIPS": 0.17}]},
    ]
    rows = aggregate_pareto_points(raw)
    assert [(r["arm"], r["anchors_mean"]) for r in rows] == [
        ("demand", 50.0), ("demand", 100.0), ("uniform", 110.0)]
    assert rows[0]["PSNR_mean"] == 27.0
    assert rows[0]["n"] == 2
    assert abs(rows[2]["PSNR_mean"] - 28.2) < 1e-9


def test_skips_points_with_no_data():
    raw = [{"arm": "demand", "factor": 0.25, "anchors": [], "metrics": []}]
    assert aggregate_pareto_points(raw) == []
