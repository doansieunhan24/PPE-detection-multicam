from eval.run_c2 import pool_results


def _r(tp, fp, fn, dur, unk=0.0, ttd=None):
    return {"tp": tp, "fp": fp, "fn": fn, "duration_s": dur, "unknown_frac": unk,
            "mean_ttd_s": ttd}  # fmt: skip


def test_pool_results_sums_held_out_counts_per_method_and_camera():
    f1 = {"P": {**_r(2, 1, 0, 3600, 0.5, 4.0), "by_cam": {"cam1": _r(2, 1, 0, 3600)}}}
    f2 = {"P": {**_r(1, 0, 1, 3600, 0.0, 1.0), "by_cam": {"cam2": _r(1, 0, 1, 3600)}}}
    p = pool_results([f1, f2])["P"]
    assert (p["tp"], p["fp"], p["fn"]) == (3, 1, 1)
    assert p["fa_per_hour"] == 0.5 and p["unknown_frac"] == 0.25
    assert p["mean_ttd_s"] == 3.0  # weighted by TP
    assert set(p["by_cam"]) == {"cam1", "cam2"} and p["by_cam"]["cam2"]["fn"] == 1
