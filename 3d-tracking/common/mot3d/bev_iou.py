"""
Bird's-eye view IoU for oriented 3D bounding boxes.
No external dependencies beyond numpy.
"""
from __future__ import annotations

import numpy as np


def bev_corners(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    """Return the 4 BEV corners of an oriented box as (4, 2) float64 array."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2.0, w / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]], dtype=np.float64)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    return local @ rot.T + np.array([x, y], dtype=np.float64)


def _shoelace_area(poly: list) -> float:
    n = len(poly)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(s) * 0.5


def _clip(polygon: list, a, b) -> list:
    """Sutherland-Hodgman: clip polygon to the left half-plane of directed edge a→b."""
    def inside(p):
        return (b[0] - a[0]) * (p[1] - a[1]) >= (b[1] - a[1]) * (p[0] - a[0])

    def intersect(p, q):
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        px, py = p[0], p[1]
        qx, qy = q[0], q[1]
        denom = (bx - ax) * (py - qy) - (by - ay) * (px - qx)
        if abs(denom) < 1e-12:
            return p
        t = ((px - ax) * (py - qy) - (py - ay) * (px - qx)) / denom
        return [ax + t * (bx - ax), ay + t * (by - ay)]

    out = []
    n = len(polygon)
    for i in range(n):
        curr, prev = polygon[i], polygon[(i - 1) % n]
        if inside(curr):
            if not inside(prev):
                out.append(intersect(prev, curr))
            out.append(curr)
        elif inside(prev):
            out.append(intersect(prev, curr))
    return out


def _intersection_area(c1: np.ndarray, c2: np.ndarray) -> float:
    poly = [list(p) for p in c2]
    n = len(c1)
    for i in range(n):
        poly = _clip(poly, c1[i], c1[(i + 1) % n])
        if not poly:
            return 0.0
    return _shoelace_area(poly) if len(poly) >= 3 else 0.0


def bev_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
    """BEV IoU between two oriented boxes [x, y, z, l, w, h, yaw]."""
    x1, y1, _, l1, w1, _, yaw1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]), float(box1[4]), float(box1[5]), float(box1[6])
    x2, y2, _, l2, w2, _, yaw2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]), float(box2[4]), float(box2[5]), float(box2[6])
    c1 = bev_corners(x1, y1, l1, w1, yaw1)
    c2 = bev_corners(x2, y2, l2, w2, yaw2)
    inter = _intersection_area(c1, c2)
    union = l1 * w1 + l2 * w2 - inter
    return float(inter / union) if union > 1e-6 else 0.0


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute N×M BEV IoU matrix between two sets of 7-param boxes."""
    n, m = len(boxes1), len(boxes2)
    mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            mat[i, j] = bev_iou_single(boxes1[i], boxes2[j])
    return mat


def dist_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return (1 - IoU) cost matrix for use with scipy.optimize.linear_sum_assignment."""
    return 1.0 - iou_matrix(boxes1, boxes2)
