"""Train the learned fall variant (§7.7, fall.method: seq) on UR Fall + MCFD skeleton JSONL.

Training data = the *val* splits used for the rule (odd sequence numbers of both datasets,
every camera); the even (test) sequences are never seen, so eval/eval_fall.py --split test
compares rule and seq fairly. Windows of seq_len frames (both datasets are 30 fps); a window
is positive when it ends in [fall start + pos_from_s, min(lying end, fall start + pos_to_s)].
Seed fixed (§13). Inputs come from tools/urfall_prepare.py and tools/mcfd_prepare.py.

    python -m training.fall.train_seq --out models/fall_seq.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.eval_fall import gt_events, in_split, single_actor
from src.behavior.seq_features import build_model, flip, frame_features, window
from src.common.config import DEFAULT_CONFIG_DIR, load_config
from src.common.jsonl import read_frames
from src.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def sequence_windows(
    path: Path, events: list[list[int]], seq_len: int, stride: int, pos: tuple[float, float]
) -> tuple[list[np.ndarray], list[int]]:
    frames = sorted(read_frames(path), key=lambda f: f.ts_ms)
    feats, ts, prev = [], [], None
    for fr in frames:
        p = single_actor(fr).persons[0] if fr.persons else None
        feats.append(frame_features(p, prev))
        ts.append(fr.ts_ms)
        prev = p
    xs, ys = [], []
    for i in range(seq_len // 2, len(feats), stride):
        t = ts[i]
        y = any(a + pos[0] * 1000 <= t <= min(b, a + pos[1] * 1000) for a, b in events)
        xs.append(window(feats[max(0, i + 1 - seq_len) : i + 1], seq_len))
        ys.append(int(y))
    return xs, ys


def load_split(roots: list[Path], split: str, seq_len: int, stride: int, pos) -> tuple:
    X, Y, groups = [], [], []
    for root in roots:
        gt = json.loads((root / "gt.json").read_text(encoding="utf-8"))
        for seq, g in gt.items():
            if not in_split(seq, split):
                continue
            for path in sorted((root / "jsonl" / seq).glob("*.jsonl")):
                xs, ys = sequence_windows(path, gt_events(g), seq_len, stride, pos)
                X += xs
                Y += ys
                groups += [f"{root.name}/{seq}"] * len(xs)
    return np.stack(X), np.array(Y, dtype=np.float32), groups


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[Path("data/external/urfall"), Path("data/external/mcfd")],
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--split", default="val", help="sequences to train on (val = odd)")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pos", type=float, nargs=2, default=[0.5, 3.0], metavar=("FROM_S", "TO_S"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models/fall_seq.pt"))
    args = ap.parse_args(argv)
    setup_logging()

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    seq_len = load_config("behavior", Path(args.config))["fall"]["seq_len"]
    X, Y, groups = load_split(args.roots, args.split, seq_len, args.stride, tuple(args.pos))
    log.info(
        "%d windows from %d sequence-cams, %d positive (%.1f %%)",
        len(Y),
        len(set(groups)),
        int(Y.sum()),
        100 * Y.mean(),
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor((1 - Y.mean()) / max(Y.mean(), 1e-6), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(Y))
        tot = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            xb = Xt[idx].numpy()
            flip_mask = rng.random(len(idx)) < 0.5
            if flip_mask.any():
                xb[flip_mask] = np.stack([flip(w) for w in xb[flip_mask]])
            xb = torch.from_numpy(xb).to(dev)
            loss = loss_fn(model(xb), Yt[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 4 or ep == args.epochs - 1:
            log.info("epoch %d loss %.4f", ep + 1, tot / len(order))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "seq_len": seq_len,
            "train": {k: str(v) for k, v in vars(args).items()},
            "n_windows": len(Y),
            "n_pos": int(Y.sum()),
        },
        args.out,
    )
    log.info("-> %s", args.out)


if __name__ == "__main__":
    main()
