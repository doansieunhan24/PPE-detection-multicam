"""C2 eval chain: per-type/association tables, val tuning, camera-count experiment."""

import json

from eval import run_baselines, run_num_cams, tune_c2
from tests.test_run_baselines import _setup


def test_run_baselines_tables_and_options(tmp_path):
    cfg, sess, gt = _setup(tmp_path)
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"overrides": {"P": {"alert.T_alert_s": 2.0}}}), "utf-8")
    out = tmp_path / "report"
    run_baselines.main(["--sessions", str(sess), "--gt", str(gt), "--out", str(out), "--config",
                        str(cfg), "--single-cam-all", "--tuned", str(tuned)])  # fmt: skip
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))["results"]
    assert "B1@cam2" in res and "B0@cam2" in res
    assert res["B1@cam2"]["overrides"]["baselines.primary_cam"] == "cam2"
    assert res["P"]["overrides"] == {"alert.T_alert_s": 2.0} and res["B2"]["overrides"] == {}
    assert res["P"]["per_type"]["PPE_HELMET"]["tp"] == 1
    assert res["P"]["per_type"]["PPE_VEST"]["n_gt"] == 0
    a = res["P"]["association"]
    assert a["pair_f1"] == 1.0 and a["frag"] == 1.0  # one person seen by both cams
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "Behavior" in md and "Association" in md and "PPE_HELMET" in md

    out2 = tmp_path / "report_cam2"
    run_baselines.main(["--sessions", str(sess), "--gt", str(gt), "--out", str(out2),
                        "--config", str(cfg), "--cams", "cam2"])  # fmt: skip
    res2 = json.loads((out2 / "results.json").read_text(encoding="utf-8"))["results"]
    assert res2["B1"]["overrides"]["baselines.primary_cam"] == "cam2"  # primary cam1 not in set
    run = json.loads((out2 / "runs" / "s01" / "P" / "run.json").read_text(encoding="utf-8"))
    assert run["cams"] == ["cam2"]


def test_tune_and_num_cams(tmp_path):
    cfg, sess, gt = _setup(tmp_path)
    space = {"P": {"alert.T_alert_s": [1.0, 3.0]}, "B1": {"alert.T_alert_s": [1.0, 3.0]}}
    res = tune_c2.tune([str(sess)], str(gt), tmp_path / "tune", str(cfg), ["P", "B1"],
                       rounds=2, space=space)  # fmt: skip
    assert set(res["overrides"]) == {"P", "B1"}
    assert res["val_metrics"]["P"]["f1"] == 1.0
    assert all(t["key"] == "alert.T_alert_s" for t in res["trace"])

    out = tmp_path / "num_cams"
    run_num_cams.main(["--sessions", str(sess), "--gt", str(gt), "--out", str(out),
                       "--config", str(cfg), "--policies", "P", "B1"])  # fmt: skip
    r = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {(x["k"], x["policy"]) for x in r["by_k"]} == {(1, "P"), (1, "B1"), (2, "P"), (2, "B1")}
    assert len(r["rows"]) == 2 * 3  # subsets {cam1}, {cam2}, {cam1, cam2} x 2 policies
    assert (out / "num_cams.png").exists()


def test_score_objectives():
    a = {"f1": 0.8, "fa_per_hour": 10.0, "recall": 0.95}
    b = {"f1": 0.8, "fa_per_hour": 5.0, "recall": 0.85}
    assert tune_c2.score(b, "f1", 0.9) > tune_c2.score(a, "f1", 0.9)  # tie on F1 → fewer FA
    assert tune_c2.score(a, "fa@recall", 0.9) > tune_c2.score(b, "fa@recall", 0.9)


def test_build_v_dataset_c2(tmp_path):
    import csv

    from training.visibility.build_v_dataset_c2 import build, label_at

    assert label_at([(10_000, 20_000)], 15_000, 1000) == 0
    assert label_at([(10_000, 20_000)], 25_000, 1000) == 1
    assert label_at([(10_000, 20_000)], 10_500, 1000) is None
    cfg, sess, gt = _setup(tmp_path)
    out = tmp_path / "v.csv"
    stats = build([sess], gt, out, cfg, stride=5)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    head = [r for r in rows if r["part"] == "head"]
    assert stats["head_worn"] > 0 and stats["head_viol"] > 0 and stats["torso_viol"] == 0
    # cam2 sees the helmet while worn (0–10 s); cam1 (back view) never does
    det = {(cam, y): {r["has_det"] for r in head if r["image"].split("/")[1] == cam
                      and r["gt"] == y} for cam in ("cam1", "cam2") for y in "01"}  # fmt: skip
    assert det[("cam2", "1")] == {"1"} and det[("cam1", "1")] == {"0"}
    assert det[("cam2", "0")] == {"0"}


def test_run_c2_check_and_report(tmp_path):
    import shutil

    import yaml

    from eval import run_c2

    cfg, sess, gt = _setup(tmp_path)
    (gt / "s01" / "intervals.csv").write_text(  # last frame is at 19960 ms
        "gt_id,type,t_start_ms,t_end_ms,note\n1,PPE_HELMET,10000,19960,\n", encoding="utf-8"
    )
    shutil.copytree(sess, sess.parent / "s02")
    shutil.copytree(gt / "s01", gt / "s02")
    split_yaml = tmp_path / "split.yaml"
    split_yaml.write_text(yaml.safe_dump({"jsonl_root": str(sess.parent), "gt_root": str(gt),
                                          "val": ["s01"], "test": ["s02"]}), "utf-8")  # fmt: skip
    split = run_c2.load_split(split_yaml)
    assert run_c2.check(split, cfg) == []
    bad = {**split, "test": [sess, sess.parent / "s03"]}
    errs = run_c2.check(bad, cfg)
    assert any("both val and test" in e for e in errs) and any("s03" in e for e in errs)
    (gt / "s02" / "intervals.csv").write_text(
        "gt_id,type,t_start_ms,t_end_ms,note\n9,PPE_HELMET,1000,2000,\n", encoding="utf-8"
    )
    assert any("label validation" in e for e in run_c2.check(split, cfg))
    out = tmp_path / "c2"
    run_c2.main(["--split", str(split_yaml), "--out", str(out), "--config", str(cfg),
                 "--steps", "report"])  # fmt: skip
    rep = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "test (reported): s02" in rep and "_(missing:" in rep


def test_rehearsal_merge_runs():
    from tools.wildtrack_rehearsal import merge_runs

    assert merge_runs([0, 500, 1000, 6000, 6500, 20000], 2000, 400) == [(0, 1000), (6000, 6500)]


def test_obs_cache_gives_identical_output(tmp_path):
    from src.analytics_runner import run

    cfg, sess, _ = _setup(tmp_path)

    def outputs(d):
        alerts = [
            {k: v for k, v in json.loads(ln).items() if k != "alert_id"}
            for ln in (d / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return (d / "states.jsonl").read_text(encoding="utf-8"), alerts

    cache = tmp_path / "cache"
    for pol, ov in (("P", {}), ("B2", {"alert.T_alert_s": 1.0}), ("P", {"evidence.tau_s": 0.5})):
        ref = run(sess, tmp_path / "ref" / pol, cfg, pol, {}, fusion_overrides=ov)
        built = run(sess, tmp_path / "c1" / pol, cfg, pol, {}, fusion_overrides=ov,
                    obs_cache=cache)  # fmt: skip
        hit = run(sess, tmp_path / "c2" / pol, cfg, pol, {}, fusion_overrides=ov,
                  obs_cache=cache)  # fmt: skip
        assert hit["obs_cache"] == "hit" and ref["groups"] == hit["groups"]
        assert outputs(tmp_path / "ref" / pol) == outputs(tmp_path / "c1" / pol)
        assert outputs(tmp_path / "ref" / pol) == outputs(tmp_path / "c2" / pol)
        assert built["obs_cache"] in ("built", "hit")
    assert len(list(cache.glob("*.pkl"))) == 1  # thresholds/policy do not change the key
    run(sess, tmp_path / "c3", cfg, "P", {}, fusion_overrides={"rois.head_expand": 2.0},
        obs_cache=cache)  # fmt: skip
    assert len(list(cache.glob("*.pkl"))) == 2  # an observation parameter does
