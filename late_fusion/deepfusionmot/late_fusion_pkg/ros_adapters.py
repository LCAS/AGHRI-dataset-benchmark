"""Conversions between standard ROS vision messages and typed detections."""

from __future__ import annotations

import math

from late_fusion_pkg.detection_types import Detection2D, Detection3D


def _yaw_from_quaternion(q) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def detections2d_from_msg(msg) -> list[Detection2D]:
    out = []
    stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
    for index, det in enumerate(msg.detections):
        cx = float(det.bbox.center.position.x)
        cy = float(det.bbox.center.position.y)
        sx = float(det.bbox.size_x)
        sy = float(det.bbox.size_y)
        score = 1.0
        label_id = -1
        label_name = ""
        if det.results:
            hyp = det.results[0].hypothesis
            score = float(hyp.score)
            label_name = str(hyp.class_id)
            try:
                label_id = int(hyp.class_id)
            except ValueError:
                label_id = -1
        out.append(
            Detection2D(
                bbox_xyxy=(cx - sx / 2.0, cy - sy / 2.0, cx + sx / 2.0, cy + sy / 2.0),
                score=score,
                label_id=label_id,
                label_name=label_name,
                detection_id=str(getattr(det, "id", index)),
                timestamp=stamp,
            )
        )
    return out


def detections3d_from_msg(msg, default_origin: str = "bottom_center") -> list[Detection3D]:
    out = []
    stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
    for index, det in enumerate(msg.detections):
        score = 1.0
        label_id = -1
        label_name = ""
        if det.results:
            hyp = det.results[0].hypothesis
            score = float(hyp.score)
            label_name = str(hyp.class_id)
            try:
                label_id = int(hyp.class_id)
            except ValueError:
                label_id = -1
        size_lwh = (
            float(det.bbox.size.x),
            float(det.bbox.size.y),
            float(det.bbox.size.z),
        )
        center_z = float(det.bbox.center.position.z)
        if default_origin == "bottom_center":
            z = center_z - size_lwh[2] / 2.0
        elif default_origin == "center":
            z = center_z
        else:
            raise ValueError(f"Unsupported box origin: {default_origin}")
        out.append(
            Detection3D(
                center_xyz=(
                    float(det.bbox.center.position.x),
                    float(det.bbox.center.position.y),
                    z,
                ),
                size_lwh=size_lwh,
                yaw=_yaw_from_quaternion(det.bbox.center.orientation),
                score=score,
                label_id=label_id,
                label_name=label_name,
                detection_id=str(getattr(det, "id", index)),
                timestamp=stamp,
                frame_id=msg.header.frame_id,
                box_origin=default_origin,
            )
        )
    return out


def detection2d_to_msg(det: Detection2D, msg_cls, bbox_cls, hyp_cls):
    msg = msg_cls()
    msg.id = det.detection_id
    x1, y1, x2, y2 = det.bbox_xyxy
    bbox = bbox_cls()
    bbox.center.position.x = float((x1 + x2) / 2.0)
    bbox.center.position.y = float((y1 + y2) / 2.0)
    bbox.center.theta = 0.0
    bbox.size_x = float(x2 - x1)
    bbox.size_y = float(y2 - y1)
    msg.bbox = bbox
    hyp = hyp_cls()
    hyp.hypothesis.class_id = str(det.label_id)
    hyp.hypothesis.score = float(det.score)
    msg.results.append(hyp)
    return msg


def detection3d_to_msg(det: Detection3D, msg_cls, bbox_cls, hyp_cls):
    msg = msg_cls()
    msg.id = det.detection_id
    bbox = bbox_cls()
    bbox.center.position.x = float(det.center_xyz[0])
    bbox.center.position.y = float(det.center_xyz[1])
    if det.box_origin == "bottom_center":
        center_z = float(det.center_xyz[2]) + float(det.size_lwh[2]) / 2.0
    elif det.box_origin == "center":
        center_z = float(det.center_xyz[2])
    else:
        raise ValueError(f"Unsupported box origin: {det.box_origin}")
    bbox.center.position.z = center_z
    bbox.center.orientation.z = math.sin(float(det.yaw) / 2.0)
    bbox.center.orientation.w = math.cos(float(det.yaw) / 2.0)
    bbox.size.x = float(det.size_lwh[0])
    bbox.size.y = float(det.size_lwh[1])
    bbox.size.z = float(det.size_lwh[2])
    msg.bbox = bbox
    hyp = hyp_cls()
    hyp.hypothesis.class_id = str(det.label_id)
    hyp.hypothesis.score = float(det.score)
    msg.results.append(hyp)
    return msg
