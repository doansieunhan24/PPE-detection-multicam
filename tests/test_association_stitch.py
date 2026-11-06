from src.association.associator import Associator
from src.common.schemas import PerceptionFrame, PersonDet


def _fr(ts, persons):
    return PerceptionFrame(cam_id="c1", frame_idx=ts // 40, ts_ms=ts, img_w=1920, img_h=1080,
                           persons=[PersonDet(track_id=t, bbox=list(b), conf=0.9)
                                    for t, b in persons])  # fmt: skip


def test_no_stitch_by_default_each_track_its_own_id():
    a = Associator({}, mode="none")
    g1 = a.step({"c1": _fr(0, [(1, (100, 100, 200, 400))])}, 0)
    g2 = a.step({"c1": _fr(500, [(2, (110, 100, 210, 400))])}, 500)
    assert g1[("c1", 1)] != g2[("c1", 2)]


def test_stitch_new_track_near_a_just_lost_one_keeps_its_id():
    a = Associator({}, mode="none", stitch_gap_ms=1500, stitch_dist_h=0.6)
    g1 = a.step({"c1": _fr(0, [(1, (100, 100, 200, 400))])}, 0)
    g2 = a.step({"c1": _fr(500, [(2, (130, 100, 230, 400))])}, 500)  # 30 px away, h 300
    assert g2[("c1", 2)] == g1[("c1", 1)]
    far = a.step({"c1": _fr(1000, [(3, (1500, 100, 1600, 400))])}, 1000)  # far: new person
    assert far[("c1", 3)] != g1[("c1", 1)]
    late = a.step({"c1": _fr(5000, [(4, (130, 100, 230, 400))])}, 5000)  # gap too long
    assert late[("c1", 4)] not in (g1[("c1", 1)], far[("c1", 3)])


def test_stitch_never_takes_the_id_of_a_track_still_visible():
    a = Associator({}, mode="none", stitch_gap_ms=1500)
    g = a.step({"c1": _fr(0, [(1, (100, 100, 200, 400))])}, 0)
    g2 = a.step({"c1": _fr(40, [(1, (100, 100, 200, 400)), (2, (120, 100, 220, 400))])}, 40)
    assert g2[("c1", 2)] != g[("c1", 1)]
