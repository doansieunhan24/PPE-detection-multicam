"""Data schemas — single source of truth (CLAUDE.md §6).

Units: image coords in original-resolution pixels, floor coords in metres, time in ms (int).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

NUM_COCO_KPTS = 17

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
Keypoint = tuple[float, float, float]  # x, y, conf


# --- 6.1 PerceptionFrame --------------------------------------------------------------


class PPEDet(BaseModel):
    cls: str
    bbox: BBox
    conf: float = Field(ge=0.0, le=1.0)


class PersonDet(BaseModel):
    track_id: int
    bbox: BBox
    conf: float = Field(ge=0.0, le=1.0)
    kpts: list[Keypoint] = Field(default_factory=list)
    ppe_dets: list[PPEDet] = Field(default_factory=list)
    truncated_bottom: bool = False

    @field_validator("kpts")
    @classmethod
    def _check_kpts(cls, v: list[Keypoint]) -> list[Keypoint]:
        if v and len(v) != NUM_COCO_KPTS:
            raise ValueError(f"expected {NUM_COCO_KPTS} COCO keypoints, got {len(v)}")
        return v


class PerceptionFrame(BaseModel):
    """One JSONL line per camera per frame."""

    cam_id: str
    frame_idx: int
    ts_ms: int
    img_w: int
    img_h: int
    persons: list[PersonDet] = Field(default_factory=list)


# --- 6.2 PersonPartState ---------------------------------------------------------------


class PartName(str, Enum):
    HEAD = "head"
    TORSO = "torso"
    FEET = "feet"


class ComplianceState(str, Enum):
    COMPLIANT = "COMPLIANT"
    VIOLATION = "VIOLATION"
    UNKNOWN = "UNKNOWN"


class CamEvidence(BaseModel):
    p: float = Field(ge=0.0, le=1.0)
    v: float = Field(ge=0.0, le=1.0)


class PartState(BaseModel):
    item: str
    state: ComplianceState = ComplianceState.UNKNOWN
    score: float = 0.0  # m_t: weighted mean log-odds
    obs: float = 0.0  # Omega_t: accumulated visibility
    per_cam: dict[str, CamEvidence] = Field(default_factory=dict)


class PersonPartState(BaseModel):
    global_id: int
    ts_ms: int
    floor_xy: tuple[float, float] | None = None  # metres
    parts: dict[PartName, PartState] = Field(default_factory=dict)
    zone: str | None = None


# --- 6.3 AlertEvent --------------------------------------------------------------------


class AlertType(str, Enum):
    PPE_HELMET = "PPE_HELMET"
    PPE_VEST = "PPE_VEST"
    PPE_BOOTS = "PPE_BOOTS"
    INTRUSION = "INTRUSION"
    FALL = "FALL"
    IMMOBILE = "IMMOBILE"
    CROWD = "CROWD"
    UNVERIFIABLE = "UNVERIFIABLE"


class Evidence(BaseModel):
    clip: str | None = None
    snapshot: str | None = None


class AlertEvent(BaseModel):
    alert_id: str
    type: AlertType
    level: int = Field(ge=1, le=3)
    global_id: int | None = None
    zone: str | None = None
    t_start_ms: int
    t_raise_ms: int
    cams: list[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
