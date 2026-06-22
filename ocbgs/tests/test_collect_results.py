import json
import os
import sys
import tempfile
import pytest

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import collect_results


def _summary_json(metrics_dict):
    """Build a summary dict matching cmd_metrics output schema."""
    return {"summary": {cp: {m: {"mean": v, "stdev": 0.0, "n": 1}
                              for m, v in per_m.items()}
                        for cp, per_m in metrics_dict.items()}}


def _write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestTableSubcommand:

    def test_two_arms_single_checkpoint(self, tmpdir, capsys):
        a = _summary_json({"ours_30000": {"PSNR": 24.18, "SSIM": 0.96, "LPIPS": 0.035}})
        b = _summary_json({"ours_30000": {"PSNR": 24.40, "SSIM": 0.97, "LPIPS": 0.030}})
        pa = os.path.join(tmpdir, "a.json")
        pb = os.path.join(tmpdir, "b.json")
        _write_json(pa, a)
        _write_json(pb, b)

        args = collect_results._parse_table_args([
            "table",
            "--arm", f"baseline={pa}",
            "--arm", f"matched={pb}",
        ])
        collect_results.cmd_table(args)

        out = capsys.readouterr().out
        assert "baseline" in out
        assert "matched" in out
        assert "24.1800" in out
        assert "24.4000" in out
        assert "+0.2200" in out

    def test_three_arms_with_baseline_label(self, tmpdir, capsys):
        a = _summary_json({"ours_30000": {"PSNR": 24.18, "SSIM": 0.96, "LPIPS": 0.035}})
        b = _summary_json({"ours_30000": {"PSNR": 24.40, "SSIM": 0.97, "LPIPS": 0.030}})
        c = _summary_json({"ours_30000": {"PSNR": 24.34, "SSIM": 0.965, "LPIPS": 0.032}})
        pa = os.path.join(tmpdir, "a.json")
        pb = os.path.join(tmpdir, "b.json")
        pc = os.path.join(tmpdir, "c.json")
        _write_json(pa, a)
        _write_json(pb, b)
        _write_json(pc, c)

        args = collect_results._parse_table_args([
            "table",
            "--arm", f"matched={pb}",
            "--arm", f"natural={pc}",
            "--arm", f"baseline={pa}",
            "--baseline-label", "baseline",
        ])
        collect_results.cmd_table(args)

        out = capsys.readouterr().out
        assert "matched" in out
        assert "natural" in out
        assert "baseline" in out
        assert "+0.2200" in out
        assert "+0.1600" in out

    def test_different_metrics_columns(self, tmpdir, capsys):
        a = _summary_json({"ours_30000": {"PSNR": 24.18, "SSIM": 0.96, "LPIPS": 0.035}})
        b = _summary_json({"ours_30000": {"PSNR": 24.40, "SSIM": 0.97, "LPIPS": 0.030}})
        pa = os.path.join(tmpdir, "a.json")
        pb = os.path.join(tmpdir, "b.json")
        _write_json(pa, a)
        _write_json(pb, b)

        args = collect_results._parse_table_args([
            "table",
            "--arm", f"baseline={pa}",
            "--arm", f"matched={pb}",
            "--metrics", "PSNR", "SSIM",
        ])
        collect_results.cmd_table(args)

        out = capsys.readouterr().out
        assert "24.1800" in out
        assert "0.9600" in out
        assert "LPIPS" not in out

    def test_missing_checkpoint_in_one_arm(self, tmpdir, capsys):
        a = _summary_json({"ours_30000": {"PSNR": 24.18, "SSIM": 0.96, "LPIPS": 0.035}})
        b = _summary_json({"ours_25000": {"PSNR": 23.10, "SSIM": 0.95, "LPIPS": 0.040}})
        pa = os.path.join(tmpdir, "a.json")
        pb = os.path.join(tmpdir, "b.json")
        _write_json(pa, a)
        _write_json(pb, b)

        args = collect_results._parse_table_args([
            "table",
            "--arm", f"baseline={pa}",
            "--arm", f"other={pb}",
        ])
        collect_results.cmd_table(args)

        out = capsys.readouterr().out
        assert "ours_30000" in out
        assert "ours_25000" in out

    def test_empty_arms_graceful(self, tmpdir):
        with pytest.raises(SystemExit):
            collect_results._parse_table_args(["table"])

    def test_no_shared_checkpoints(self, tmpdir, capsys):
        a = _summary_json({"ours_30000": {"PSNR": 24.18, "SSIM": 0.96, "LPIPS": 0.035}})
        b = _summary_json({"ours_30000": {"PSNR": 24.40, "SSIM": 0.97, "LPIPS": 0.030}})
        pa = os.path.join(tmpdir, "a.json")
        pb = os.path.join(tmpdir, "b.json")
        _write_json(pa, a)
        _write_json(pb, b)

        sys.argv = ["table", "--arm", f"baseline={pa}", "--arm", f"other={pb}"]
        args = collect_results._parse_table_args([
            "table",
            "--arm", f"baseline={pa}",
            "--arm", f"other={pb}",
        ])
        collect_results.cmd_table(args)

        out = capsys.readouterr().out
        assert "ours_30000" in out
