import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "plot_pareto",
    os.path.join(os.path.dirname(__file__), "plot_pareto.py"),
)
plot_pareto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_pareto)


def test_split_curves_groups_by_arm_sorted_by_anchors():
    rows = [
        {"arm": "demand", "anchors": 100.0, "PSNR": 28.5},
        {"arm": "uniform", "anchors": 110.0, "PSNR": 28.2},
        {"arm": "demand", "anchors": 50.0, "PSNR": 27.0},
    ]
    curves = plot_pareto.split_curves(rows, metric="PSNR")
    assert set(curves.keys()) == {"demand", "uniform"}
    xs, ys = curves["demand"]
    assert xs == [50.0, 100.0]
    assert ys == [27.0, 28.5]


def test_load_pareto_rows_parses_csv(tmp_path):
    csv = tmp_path / "pareto.csv"
    csv.write_text(
        "arm,factor,anchors,PSNR,SSIM,LPIPS,n\n"
        "demand,1,100,28.5000,0.8300,0.1700,3\n"
    )
    rows = plot_pareto.load_pareto_rows(str(csv))
    assert rows[0]["arm"] == "demand"
    assert rows[0]["anchors"] == 100.0
    assert rows[0]["PSNR"] == 28.5
