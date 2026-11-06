"""One person, one part: v, p and B1 vs P state over time, with their alerts (from eval runs)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, sys.argv[1])
from eval.run_baselines import gid_to_gt

runs, take, cam, gt_id, part, out = (
    Path(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
    int(sys.argv[5]),
    sys.argv[6],
    sys.argv[7],
)
item = {"head": "PPE_HELMET", "torso": "PPE_VEST"}[part]
INK, MUTED, GRID = "#1F2A33", "#5B6770", "#D9DEE2"
COL = {"COMPLIANT": "#0E6E6E", "VIOLATION": "#E07A1F", "UNKNOWN": "#C9D1D6"}
plt.rcParams.update({"font.family": "Arial", "font.size": 11})


def series(policy):
    d = runs / policy
    mp = gid_to_gt(d, Path(sys.argv[1]) / "data/annotations" / take / "tracks.csv")
    t, v, p, st = [], [], [], []
    for ln in open(d / "states.jsonl"):
        s = json.loads(ln)
        if mp.at(s["global_id"], s["ts_ms"]) != gt_id:
            continue
        ps = s["parts"][part]
        pc = next(iter(ps["per_cam"].values()), None)
        if pc is None:
            continue
        t.append(s["ts_ms"] / 1000)
        v.append(pc["v"])
        p.append(pc["p"])
        st.append(ps["state"])
    al = [json.loads(ln) for ln in open(d / "alerts.jsonl")]
    al = [
        a["t_raise_ms"] / 1000
        for a in al
        if a["type"] == item
        and a["global_id"] is not None
        and mp.at(a["global_id"], a["t_raise_ms"]) == gt_id
    ]
    return t, v, p, st, al


tP, vP, pP, sP, aP = series("P")
tB, _, _, sB, aB = series("B1")
fig, ax = plt.subplots(
    3, 1, figsize=(10, 4.8), dpi=200, sharex=True, gridspec_kw={"height_ratios": [2, 0.6, 0.6]}
)
ax[0].plot(tP, vP, color="#0E6E6E", lw=1.2, label="v (độ quan sát)")
ax[0].plot(tP, pP, color="#9AA5AD", lw=1.0, label="p (xác suất có PPE)")
ax[0].set_ylim(0, 1.05)
ax[0].legend(loc="upper right", frameon=False, ncol=2)
ax[0].spines[["top", "right"]].set_visible(False)
ax[0].yaxis.grid(True, color=GRID)
for _k, (a, t, st, al, name) in enumerate([(ax[1], tB, sB, aB, "B1"), (ax[2], tP, sP, aP, "P")]):
    for i in range(len(t) - 1):
        if t[i + 1] - t[i] < 0.5:
            a.axvspan(t[i], t[i + 1], color=COL.get(st[i], "#FFFFFF"), lw=0)
    for x in al:
        a.plot([x], [0.5], marker="v", color=INK, ms=9)
    a.set_yticks([])
    a.set_ylabel(name, rotation=0, ha="right", va="center", color=INK, fontsize=12)
    for sp in a.spines.values():
        sp.set_visible(False)
ax[2].set_xlabel("giây")

h = [Patch(color=COL[k], label=k) for k in COL] + [
    Line2D([], [], marker="v", ls="", color=INK, label="cảnh báo")
]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(out)
print("alerts B1", aB, "P", aP)
