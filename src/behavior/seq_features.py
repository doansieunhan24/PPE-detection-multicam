"""Per-frame skeleton features and the 1D temporal CNN for the learned fall variant (§7.7).

Shared by training/fall/train_seq.py and BehaviorEngine (fall.method: seq), so both see the
same features. Keypoints are normalized by the person bbox (centre, height), which makes the
features independent of image size and camera distance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.common.schemas import NUM_COCO_KPTS, PersonDet

FEAT_DIM = NUM_COCO_KPTS * 3 + 4  # kpts (x, y, c) + aspect, v_y, dh, present
# COCO left/right pairs, for horizontal-flip augmentation
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def frame_features(p: PersonDet | None, prev: PersonDet | None) -> np.ndarray:
    """Features of one frame; zeros (present = 0) when the person is not detected."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    if p is None:
        return f
    x1, y1, x2, y2 = p.bbox
    h = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if p.kpts:
        k = np.asarray(p.kpts, dtype=np.float32).reshape(NUM_COCO_KPTS, 3)
        k[:, 0] = (k[:, 0] - cx) / h
        k[:, 1] = (k[:, 1] - cy) / h
        f[: NUM_COCO_KPTS * 3] = k.reshape(-1)
    f[-4] = (x2 - x1) / h
    if prev is not None:
        ph = max(1.0, prev.bbox[3] - prev.bbox[1])
        f[-3] = (cy - (prev.bbox[1] + prev.bbox[3]) / 2) / ph
        f[-2] = (h - ph) / ph
    f[-1] = 1.0
    return f


def window(frames: Sequence[np.ndarray], seq_len: int) -> np.ndarray:
    """Last seq_len frame features, zero-padded at the front → (FEAT_DIM, seq_len)."""
    w = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)
    tail = list(frames)[-seq_len:]
    if tail:
        w[seq_len - len(tail) :] = np.stack(tail)
    return w.T


def flip(w: np.ndarray) -> np.ndarray:
    """Mirror a (FEAT_DIM, T) window horizontally (x → −x, swap left/right joints)."""
    out = w.copy()
    k = out[: NUM_COCO_KPTS * 3].reshape(NUM_COCO_KPTS, 3, -1)
    k[:, 0] *= -1
    for a, b in FLIP_PAIRS:
        k[[a, b]] = k[[b, a]]
    return out


def build_model(hidden: int = 64, dropout: float = 0.3):
    """Small dilated temporal CNN → fall logit. torch is imported only when needed."""
    import torch
    from torch import nn

    class FallNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(FEAT_DIM, hidden, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4),
                nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.net(x)
            return self.head(torch.cat([z.mean(-1), z.amax(-1)], dim=1)).squeeze(1)

    return FallNet()
