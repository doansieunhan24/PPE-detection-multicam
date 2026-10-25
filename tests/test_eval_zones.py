from eval.eval_zones import episodes, gt_events, match, pick_zones, rect


def test_episodes_hold_semantics():
    act = [(0, True), (500, True), (1000, True), (1500, False), (2000, True), (2500, False)]
    assert episodes(act, 1000) == [(0, 1000)]  # second run lasts 0 ms
    # instants are consecutive samples: 3000 → 4500 is one 1.5 s run
    assert episodes(act + [(3000, True), (4500, True)], 1000) == [(0, 1000), (3000, 4500)]
    assert episodes([(0, True), (500, True), (1000, True)], 1000) == [(0, 1000)]  # open at end


def test_match_one_to_one():
    assert match([1000, 1100, 9000], [(500, 2000)], 500) == (1, 2, 0)
    assert match([], [(0, 1)], 0) == (0, 0, 1)


def test_gt_events_intrusion_and_crowd():
    gt = {
        ts: [
            {"gt_id": 1, "xy_m": [0.5, 0.5] if ts <= 1000 else [5.0, 5.0]},
            {"gt_id": 2, "xy_m": [0.6, 0.6]},
        ]
        for ts in range(0, 3000, 500)
    }
    zones = [
        {"id": "F1", "type": "forbidden", "polygon_m": rect(0, 0, 2.0)},
        {"id": "R1", "type": "risk", "polygon_m": rect(0, 0, 2.0), "crowd_max": 1},
    ]
    ev = gt_events(gt, zones, hold_intr_ms=1000, hold_crowd_ms=1000)
    assert sorted(ev["F1"]) == [(0, 1000), (0, 2500)]
    assert ev["R1"] == [(0, 1000)]


def test_pick_zones_deterministic():
    gt = {
        ts: [{"gt_id": g, "xy_m": [g % 4 * 1.0, ts / 1000]} for g in range(8)]
        for ts in range(0, 8000, 500)
    }
    a = pick_zones(gt, (0, 0, 8, 8), 2, 2)
    assert a == pick_zones(gt, (0, 0, 8, 8), 2, 2)
    zones, ks = a
    assert [z["type"] for z in zones] == ["forbidden", "forbidden", "risk", "risk"]
    assert set(ks) == {"R1", "R2"}
