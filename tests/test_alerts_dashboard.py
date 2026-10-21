from fastapi.testclient import TestClient

from src.alerts.notifier_telegram import TelegramNotifier, format_alert
from src.alerts.store import AlertStore
from src.common.schemas import AlertEvent, AlertType, Evidence
from src.dashboard.app import create_app, latest_states


def _alert(i, level=2, t=1000):
    return AlertEvent(
        alert_id=f"a{i}",
        type=AlertType.PPE_HELMET,
        level=level,
        global_id=5,
        zone="Z1",
        t_start_ms=t - 3000,
        t_raise_ms=t,
        cams=["cam1", "cam3"],
        evidence=Evidence(clip="c.mp4"),
    )


def test_store_roundtrip_and_ack():
    s = AlertStore(":memory:")
    s.add_many([_alert(1, 1, 1000), _alert(2, 3, 2000)])
    rows = s.recent()
    assert [r["alert_id"] for r in rows] == ["a2", "a1"]
    assert rows[0]["cams"] == ["cam1", "cam3"] and rows[0]["evidence"]["clip"] == "c.mp4"
    assert [r["alert_id"] for r in s.recent(min_level=2)] == ["a2"]
    assert s.ack("a2") and not s.ack("nope")
    assert [r["alert_id"] for r in s.recent(unacked_only=True)] == ["a1"]


def test_telegram_level_filter_and_failure_is_swallowed():
    sent = []

    def post(url, json, timeout):
        sent.append((url, json))

        class R:
            status_code = 200

        return R()

    n = TelegramNotifier(token="T", chat_id="C", min_level=2, post=post)
    assert not n.send(_alert(1, level=1))
    assert n.send(_alert(2, level=3))
    assert sent[0][0].endswith("/botT/sendMessage") and sent[0][1]["chat_id"] == "C"
    assert "PPE_HELMET" in format_alert(_alert(2)) and "Z1" in format_alert(_alert(2))

    def boom(*a, **k):
        raise OSError("network down")

    assert not TelegramNotifier(token="T", chat_id="C", post=boom).send(_alert(3))


def test_telegram_disabled_by_config():
    assert TelegramNotifier.from_config({"telegram": {"enabled": False}}) is None


def test_dashboard_api(tmp_path):
    store = AlertStore(":memory:")
    store.add(_alert(1))
    states = tmp_path / "states.jsonl"
    states.write_text(
        '{"global_id": 1, "ts_ms": 0, "floor_xy": [1, 2], "parts": {}, "zone": null}\n'
        '{"global_id": 1, "ts_ms": 5000, "floor_xy": [1.5, 2], "parts": {}, "zone": null}\n'
        '{"global_id": 2, "ts_ms": 100, "floor_xy": [3, 3], "parts": {}, "zone": null}\n',
        encoding="utf-8",
    )
    zones = {"zones": [{"id": "Z1", "type": "risk", "polygon_m": [[0, 0], [4, 0], [4, 4]]}]}
    c = TestClient(create_app(store, states, zones))
    assert c.get("/api/alerts").json()[0]["alert_id"] == "a1"
    assert c.post("/api/alerts/a1/ack").json() == {"ok": True}
    assert c.post("/api/alerts/zzz/ack").status_code == 404
    st = c.get("/api/state").json()
    assert [s["global_id"] for s in st] == [1]  # gid 2 last seen > 2 s before newest
    assert st[0]["floor_xy"] == [1.5, 2]
    assert c.get("/api/zones").json()[0]["id"] == "Z1"
    assert "Giám sát" in c.get("/").text


def test_latest_states_missing_file(tmp_path):
    assert latest_states(tmp_path / "none.jsonl") == []
