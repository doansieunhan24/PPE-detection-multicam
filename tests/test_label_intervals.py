import csv
import json

import pytest

from src.common.schemas import PerceptionFrame, PersonDet
from tools.label_intervals import convert_rows, main, parse_clock_ms, parse_gt_id


def test_parse_clock_ms():
    assert parse_clock_ms("83.5") == 83500
    assert parse_clock_ms("1:23.5") == 83500
    assert parse_clock_ms("0:01:23.5") == 83500
    assert parse_clock_ms("14:03:20") == (14 * 3600 + 3 * 60 + 20) * 1000
    for bad in ("", "1::2", "a:10", "1:2:3:4"):
        with pytest.raises(ValueError):
            parse_clock_ms(bad)


def test_parse_gt_id():
    assert parse_gt_id("P2") == 2 and parse_gt_id("p3") == 3 and parse_gt_id(" 7 ") == 7


def test_convert_rows_video_and_wall():
    rows = [{"gt_id": "P1", "type": "PPE_HELMET", "start": "1:02", "end": "1:31", "note": "V1"}]
    assert convert_rows(rows, t0_ms=500) == [[1, "PPE_HELMET", 62500, 91500, "V1", ""]]
    wall = [{"gt_id": "P1", "type": "INTRUSION", "start": "14:03:30", "end": "14:03:35"}]
    anchor = parse_clock_ms("14:03:20")
    # sync event at ref media time 7 s, ref first frame ts 0 → 10 s after it = 17 s
    want = [[1, "INTRUSION", 17000, 22000, "", ""]]
    assert convert_rows(wall, t0_ms=7000, anchor_ms=anchor) == want
    with pytest.raises(SystemExit):
        convert_rows([{"gt_id": "1", "type": "FALL", "start": "5", "end": "5"}], 0)


def _jsonl(path, cam, t0, n=100):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            p = PersonDet(track_id=1, bbox=(0, 0, 10, 20), conf=0.9)
            fr = PerceptionFrame(
                cam_id=cam, frame_idx=i, ts_ms=t0 + 40 * i, img_w=64, img_h=64, persons=[p]
            )
            f.write(fr.model_dump_json() + "\n")


def test_convert_cli_wall_clock_then_validate(tmp_path):
    jd = tmp_path / "jsonl"
    jd.mkdir()
    _jsonl(jd / "cam1.jsonl", "cam1", 1000)
    (tmp_path / "check.json").write_text(
        json.dumps({"cams": {"cam1": {"sync_start_ms": 1000}}}), encoding="utf-8"
    )
    src = tmp_path / "src.csv"
    src.write_text("gt_id,type,start,end,note\nP1,PPE_VEST,9:00:00.5,9:00:02,x\n", encoding="utf-8")
    ann = tmp_path / "ann"
    main(
        [
            "convert",
            "--jsonl",
            str(jd),
            "--src",
            str(src),
            "--out",
            str(ann / "intervals.csv"),
            "--clock",
            "wall",
            "--anchor",
            "9:00:00",
            "--session-check",
            str(tmp_path / "check.json"),
        ]
    )
    with open(ann / "intervals.csv", encoding="utf-8", newline="") as f:
        r = next(csv.DictReader(f))
    assert (r["gt_id"], r["t_start_ms"], r["t_end_ms"]) == ("1", "2500", "4000")
    (ann / "tracks.csv").write_text(
        "cam_id,track_id,first_ms,last_ms,n_frames,gt_id\ncam1,1,1000,4960,100,1\n",
        encoding="utf-8",
    )
    main(["validate", "--jsonl", str(jd), "--ann", str(ann)])


def test_convert_and_validate_cams(tmp_path):
    rows = [{"gt_id": "P1", "type": "INTRUSION", "start": "1", "end": "2", "cams": "cam3;cam2"}]
    assert convert_rows(rows, t0_ms=0)[0][-1] == "cam2 cam3"
    jd, ann = tmp_path / "jsonl", tmp_path / "ann"
    jd.mkdir()
    ann.mkdir()
    _jsonl(jd / "cam1.jsonl", "cam1", 1000)
    (ann / "tracks.csv").write_text(
        "cam_id,track_id,first_ms,last_ms,n_frames,gt_id\ncam1,1,1000,4960,100,1\n",
        encoding="utf-8",
    )
    iv = "gt_id,type,t_start_ms,t_end_ms,note,cams\n1,PPE_VEST,2000,3000,,{}\n"
    (ann / "intervals.csv").write_text(iv.format("cam1"), encoding="utf-8")
    main(["validate", "--jsonl", str(jd), "--ann", str(ann)])
    (ann / "intervals.csv").write_text(iv.format("cam9"), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["validate", "--jsonl", str(jd), "--ann", str(ann)])
