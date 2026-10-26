"""Export ONNX for DeepStream with the marcoslucianops export scripts (docker/README.md).

    python -m tools.export_ds_onnx pose -w models/ds/yolo11m-pose.pt -s 960 --dynamic
    python -m tools.export_ds_onnx det  -w models/ds/sh17chv_crop_y11s.pt -s 320 --dynamic

`pose` runs third_party/DeepStream-Yolo-Pose/utils/export_yolo11_pose.py, `det` runs
third_party/DeepStream-Yolo/utils/export_yolo11.py; extra arguments are passed through. The output
layout differs from Ultralytics' own export, so keep these ONNX files apart (models/ds/).

The scripts were written for the TorchScript exporter; torch ≥ 2.9 defaults to the dynamo
exporter, so this wrapper forces dynamo=False. labels.txt is written to the weights' folder.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "pose": ROOT / "third_party/DeepStream-Yolo-Pose/utils/export_yolo11_pose.py",
    "det": ROOT / "third_party/DeepStream-Yolo/utils/export_yolo11.py",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SCRIPTS:
        raise SystemExit(__doc__)
    script = SCRIPTS[argv.pop(0)]
    if not script.is_file():
        raise SystemExit(f"{script} not found: clone the repo into third_party/ (docker/README.md)")
    weights = Path(argv[argv.index("-w") + 1]).resolve()
    argv[argv.index("-w") + 1] = str(weights)

    export = torch.onnx.export

    def legacy_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return export(*args, **kwargs)

    torch.onnx.export = legacy_export
    os.chdir(weights.parent)  # the scripts write labels.txt into the working directory
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
