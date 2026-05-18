"""
Write 3D tracking results in MOT3D CSV format.

Format (one row per tracked object per frame):
    frame_id,track_id,x,y,z,l,w,h,yaw,score

- frame_id: 1-indexed frame number within the scene
- track_id: unique integer ID for the track within the scene
- x,y,z: box center in LiDAR coordinates
- l,w,h: box dimensions (length, width, height)
- yaw: rotation around Z-axis in radians
- score: detection confidence (1.0 for ground-truth rows)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HEADER = "frame_id,track_id,x,y,z,l,w,h,yaw,score\n"


class Mot3dWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(HEADER, encoding="utf-8")

    def write_frame(
        self,
        frame_id: int,
        tracks: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write one frame of tracks.

        tracks: (N, 8) array — columns [x, y, z, l, w, h, yaw, track_id]
        scores: optional (N,) confidence values; defaults to 1.0
        """
        if tracks is None or len(tracks) == 0:
            return
        lines = []
        for i, t in enumerate(tracks):
            x, y, z, l, w, h, yaw, tid = (
                float(t[0]), float(t[1]), float(t[2]),
                float(t[3]), float(t[4]), float(t[5]),
                float(t[6]), int(t[7]),
            )
            score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            lines.append(
                f"{frame_id},{tid},{x:.4f},{y:.4f},{z:.4f},"
                f"{l:.4f},{w:.4f},{h:.4f},{yaw:.4f},{score:.4f}\n"
            )
        with self.path.open("a", encoding="utf-8") as f:
            f.writelines(lines)


def load_mot3d(path: Path) -> dict[int, np.ndarray]:
    """
    Load a MOT3D file into a dict mapping frame_id → (N, 9) array.

    Columns of the returned array: [track_id, x, y, z, l, w, h, yaw, score]
    """
    rows: dict[int, list] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("frame_id"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            vals = [float(v) for v in parts[2:9]]
            score = float(parts[9]) if len(parts) > 9 else 1.0
            rows.setdefault(fid, []).append([tid] + vals + [score])
    return {fid: np.array(v, dtype=np.float64) for fid, v in rows.items()}
