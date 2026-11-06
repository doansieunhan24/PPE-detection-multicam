import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2]
rows = [r for r in d["rows"] if r["param"] == "alert.T_alert_s"]
plt.rcParams.update({"font.family": "Arial", "font.size": 12})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
sty = {"P": ("#0E6E6E", "o", 2.2), "B1": ("#9AA5AD", "s", 1.4), "B4": ("#5B6770", "^", 1.4)}
for ax, item, ttl in [
    (axs[0], "PPE_HELMET", "Mũ: P ngang baseline; ít báo sai hơn 8–15% ở recall 0.68–0.84"),
    (axs[1], "PPE_VEST", "Gile: P không báo lần nào (recall 0)"),
]:
    for pol, (c, m, lw) in sty.items():
        rs = sorted([r for r in rows if r["policy"] == pol], key=lambda r: r["value"])
        xs = [r["per_type"][item]["recall"] for r in rs]
        ys = [max(r["per_type"][item]["fa_per_hour"], 0.5) for r in rs]
        ax.plot(xs, ys, color=c, marker=m, lw=lw, label=pol)
        for r, x, y in zip(rs, xs, ys, strict=True):
            ax.annotate(
                f"{r['value']:g}s",
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
                color=c,
            )
    ax.set_yscale("log")
    ax.set_xlabel("Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#E3E7EA", which="both")
    ax.set_title(ttl, loc="left", fontsize=13, color="#1F2A33")
axs[0].set_ylabel("Báo sai / giờ-camera (log)")
axs[0].legend(frameon=False, loc="upper left")
axs[1].text(0.06, 1.2, "0 báo sai\n(vẽ ở 0.5 trên trục log)", fontsize=9, color="#0E6E6E")
fig.tight_layout()
fig.savefig(out)
print("ok")
