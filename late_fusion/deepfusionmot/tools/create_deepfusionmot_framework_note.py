#!/usr/bin/env python3
"""Create the AGHRI DeepFusionMOT framework note and diagrams."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "deepfusionmot_framework_note"
FIG_DIR = OUT_DIR / "figures"

W = 2000
H = 1250
BG = "#f7f9fc"
INK = "#14213d"
MUTED = "#526070"
BLUE = "#2563eb"
GREEN = "#16a34a"
CYAN = "#0891b2"
RED = "#dc2626"
YELLOW = "#ca8a04"
PURPLE = "#7c3aed"
GRAY = "#e5e7eb"
DARK_GRAY = "#374151"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ):
        path = base / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUB = font(25, True)
F_BODY = font(22)
F_SMALL = font(18)
F_TINY = font(16)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if text_width(draw, trial, fnt) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((60, 42), title, font=F_TITLE, fill=INK)
    draw.text((62, 98), subtitle, font=F_BODY, fill=MUTED)


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str = "white",
    outline: str = "#cbd5e1",
    title_color: str = INK,
) -> None:
    draw.rounded_rectangle(xy, radius=20, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 24, y1 + 22), title, font=F_SUB, fill=title_color)
    y = y1 + 62
    for line in wrap(draw, body, F_BODY, x2 - x1 - 48):
        draw.text((x1 + 24, y), line, font=F_BODY, fill=DARK_GRAY)
        y += 31


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, fill: str) -> None:
    draw.rounded_rectangle(xy, radius=28, fill=fill)
    tw = text_width(draw, text, F_SMALL)
    x1, y1, x2, y2 = xy
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + 14), text, font=F_SMALL, fill="white")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#64748b",
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        pts = [(ex, ey), (ex - 18 * sign, ey - 10), (ex - 18 * sign, ey + 10)]
    else:
        sign = 1 if dy >= 0 else -1
        pts = [(ex, ey), (ex - 10, ey - 18 * sign), (ex + 10, ey - 18 * sign)]
    draw.polygon(pts, fill=color)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def save(img: Image.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name)


def framework_overview() -> None:
    img, d = canvas()
    draw_title(d, "AGHRI DeepFusionMOT Framework", "Offline tracking layer built on the existing camera-LiDAR late-fusion repo")
    box(d, (70, 180, 390, 410), "AGHRI Sensors", "ZED RGB image\nLivox point cloud\nCameraInfo / TF\nrecording timestamps", "#eff6ff", BLUE)
    box(d, (500, 145, 900, 345), "Camera Detector", "YOLO11s generic\nYOLO11s fine-tuned\n2D person boxes + scores", "#eef2ff", BLUE)
    box(d, (500, 405, 900, 630), "LiDAR Detector", "SECOND generic/fine-tuned\nPointPillars generic/fine-tuned\n3D person boxes + scores", "#ecfdf5", GREEN)
    box(d, (1030, 255, 1435, 560), "Late Fusion", "Project LiDAR boxes into the ZED image.\nMatch YOLO and LiDAR by 2D IoU.\nProduce matched, unmatched, and ALL3D outputs.", "#f0fdfa", CYAN)
    box(d, (1545, 215, 1925, 580), "DeepFusionMOT Tracker", "Kalman prediction\nAGHRI odom compensation\nDeepFusionMOT association cascade\nTrack lifecycle and IDs", "#faf5ff", PURPLE)
    box(d, (245, 785, 640, 1035), "Offline Evaluation", "Bundled TrackEval metric family:\nHOTA, DetA, AssA\nMOTA, MOTP, IDF1, IDSW\nFPS", "white", "#94a3b8")
    box(d, (805, 785, 1200, 1035), "Artifacts", "tracking CSV/JSON/MD\nper-recording summaries\nvisual frames\nframework diagrams", "white", "#94a3b8")
    box(d, (1365, 785, 1760, 1035), "ROS 2 Path", "Current ROS 2 fusion and RViz are available.\nThe next layer is a ROS 2 DeepFusionMOT tracking node.", "white", "#94a3b8")
    arrow(d, (390, 265), (500, 235), BLUE)
    arrow(d, (390, 335), (500, 515), GREEN)
    arrow(d, (900, 245), (1030, 340), BLUE)
    arrow(d, (900, 515), (1030, 465), GREEN)
    arrow(d, (1435, 408), (1545, 408), PURPLE)
    arrow(d, (1735, 580), (1735, 785), PURPLE)
    pill(d, (710, 680, 1290, 740), "Eight AGHRI detector combinations", DARK_GRAY)
    arrow(d, (810, 740), (640, 785), "#64748b")
    arrow(d, (1000, 740), (1000, 785), "#64748b")
    save(img, "01_framework_overview.png")


def detector_fusion_inputs() -> None:
    img, d = canvas()
    draw_title(d, "Detection-Level Fusion Inputs", "The tracking stage receives the same typed detections that power offline fusion and RViz")
    box(d, (70, 185, 410, 390), "YOLO 2D", "Input: ZED RGB frame\nOutput: image boxes\nFields: xyxy, class, score, stamp\nSource: generic or fine-tuned cache/live node", "#eef2ff", BLUE)
    box(d, (70, 470, 410, 675), "3D Detector", "Input: Livox cloud\nOutput: 3D boxes\nFields: x,y,z,l,w,h,yaw, score, stamp\nSource: SECOND or PointPillars", "#ecfdf5", GREEN)
    box(d, (535, 315, 885, 540), "Calibration Projection", "T_camera_lidar from extrinsics.json\nCameraInfo.P / intrinsics.json\nProject 8 LiDAR cuboid corners into image space", "#fff7ed", YELLOW)
    box(d, (1020, 315, 1365, 540), "2D IoU Matching", "Projected 3D footprint is associated to YOLO box.\nHungarian assignment maximizes IoU under threshold gates.", "#f0fdfa", CYAN)
    box(d, (1465, 185, 1725, 390), "Matched 3D", "Camera-supported LiDAR boxes feed the first tracking association stage.", "#dcfce7", GREEN)
    box(d, (1465, 455, 1725, 650), "Unmatched", "Unmatched LiDAR boxes are preserved in ALL3D.\nUnmatched camera boxes can maintain 2D tracks.", "#fee2e2", RED)
    box(d, (560, 735, 1240, 925), "Important Boundary", "Detection-level fusion does not reshape detector geometry or learn new features. It supplies structured evidence to DeepFusionMOT: fused 3D detections, LiDAR-only 3D detections, and camera-only 2D detections.", "white", "#94a3b8")
    arrow(d, (410, 285), (535, 380), BLUE)
    arrow(d, (410, 575), (535, 475), GREEN)
    arrow(d, (885, 425), (1020, 425), CYAN)
    arrow(d, (1365, 390), (1465, 285), GREEN)
    arrow(d, (1365, 470), (1465, 550), RED)
    save(img, "02_detector_and_fusion_inputs.png")


def tracking_update_cycle() -> None:
    img, d = canvas()
    draw_title(d, "DeepFusionMOT Tracking Update Cycle", "One AGHRI recording is processed as an ordered timestamped sequence")
    boxes = [
        ((90, 200, 395, 380), "Frame Inputs", "YOLO boxes\nLiDAR boxes\nCalibration\nOptional odom"),
        ((500, 200, 805, 380), "Predict Tracks", "3D and 2D Kalman filters advance existing tracks before association."),
        ((910, 200, 1215, 380), "Ego Motion", "AGHRI odom transforms previous LiDAR-frame track states into the current LiDAR frame."),
        ((1320, 200, 1625, 380), "Associate", "DeepFusionMOT cascade matches detections to existing tracks."),
        ((500, 610, 805, 800), "Update / Create", "Matched tracks update state.\nUnmatched detections create tentative tracks."),
        ((910, 610, 1215, 800), "Mark Missed", "Unmatched tentative tracks delete immediately.\nConfirmed tracks can reactivate."),
        ((1320, 610, 1625, 800), "Output", "Confirmed 3D tracks plus early-frame tracks under strict output rule."),
    ]
    fills = ["#eff6ff", "#f8fafc", "#fff7ed", "#f0fdfa", "#ecfdf5", "#fef2f2", "#faf5ff"]
    outlines = [BLUE, "#94a3b8", YELLOW, CYAN, GREEN, RED, PURPLE]
    for (xy, title, body), fill, outline in zip(boxes, fills, outlines):
        box(d, xy, title, body, fill, outline)
    arrow(d, (395, 290), (500, 290), BLUE)
    arrow(d, (805, 290), (910, 290), YELLOW)
    arrow(d, (1215, 290), (1320, 290), CYAN)
    arrow(d, (1470, 380), (1470, 610), PURPLE)
    arrow(d, (1320, 705), (1215, 705), RED)
    arrow(d, (910, 705), (805, 705), GREEN)
    arrow(d, (652, 610), (652, 380), "#64748b")
    d.text((95, 925), "Strict AGHRI preset mirrors applicable DeepFusionMOT Pedestrian settings: min_hits=3, max_age_frames=25, 3D IoU gate=0.01, 2D IoU gate=0.50.", font=F_BODY, fill=INK)
    save(img, "03_tracking_update_cycle.png")


def association_cascade() -> None:
    img, d = canvas()
    draw_title(d, "DeepFusionMOT Association Cascade", "The AGHRI port keeps the original idea: fused evidence first, then single-modality support")
    steps = [
        ("1. Fused 3D to 3D tracks", "Matched camera-supported LiDAR boxes update active 3D tracks using BEV IoU or the configured 3D metric.", GREEN),
        ("2. LiDAR-only 3D to 3D tracks", "Unmatched LiDAR detections can still continue 3D tracks. They are not discarded from ALL3D.", CYAN),
        ("3. Camera-only 2D to 2D tracks", "Unmatched YOLO detections maintain 2D-only tracks when no usable 3D detection exists.", BLUE),
        ("4. 2D tracks support 3D tracks", "Projected 3D tracks can receive camera support through 2D IoU, improving continuity when detector outputs fluctuate.", PURPLE),
    ]
    y = 185
    for idx, (title, body, color) in enumerate(steps):
        box(d, (150, y, 760, y + 155), title, body, "white", color, color)
        box(d, (1040, y, 1650, y + 155), f"Output after stage {idx + 1}", "Matched pairs are removed from later stages; remaining tracks and detections flow to the next association gate.", "#f8fafc", "#cbd5e1")
        arrow(d, (760, y + 78), (1040, y + 78), color)
        if idx < 3:
            arrow(d, (445, y + 155), (445, y + 220), "#64748b")
            arrow(d, (1345, y + 155), (1345, y + 220), "#64748b")
        y += 220
    save(img, "04_deep_association_cascade.png")


def lifecycle() -> None:
    img, d = canvas()
    draw_title(d, "Track Lifecycle and ID Semantics", "Strict mode adopts DeepFusionMOT-style confirmation, deletion, reactivation, and ID parity")
    box(d, (90, 250, 390, 455), "Tentative", "New detection creates a tentative 3D or 2D track.\nUnmatched tentative tracks are deleted immediately in strict mode.", "#fefce8", YELLOW)
    box(d, (535, 250, 835, 455), "Confirmed", "A track is confirmed after enough hits.\nIt is eligible for normal output.", "#dcfce7", GREEN)
    box(d, (980, 250, 1280, 455), "Reactivate", "Confirmed but unmatched tracks can wait up to max_age_frames=25 before deletion.", "#eff6ff", BLUE)
    box(d, (1425, 250, 1725, 455), "Deleted", "Expired or invalid tracks are removed from active state.", "#fee2e2", RED)
    arrow(d, (390, 350), (535, 350), GREEN)
    arrow(d, (835, 350), (980, 350), BLUE)
    arrow(d, (1280, 350), (1425, 350), RED)
    arrow(d, (1130, 250), (685, 250), GREEN)
    d.text((820, 210), "matched again", font=F_SMALL, fill=GREEN)
    box(d, (210, 670, 710, 870), "ID Policy", "Strict AGHRI mode mirrors the original DeepFusionMOT convention where 3D tracks use even IDs and 2D tracks use odd IDs.", "white", "#94a3b8")
    box(d, (930, 670, 1430, 870), "Output Rule", "The strict 3D output is confirmed tracks plus early-frame tracks while frame_count <= min_hits. This avoids reporting every unstable tentative track.", "white", "#94a3b8")
    save(img, "05_track_lifecycle.png")


def ego_motion() -> None:
    img, d = canvas()
    draw_title(d, "AGHRI Ego-Motion Compensation", "The AGHRI port replaces KITTI OXTS with per-recording odom metadata")
    box(d, (90, 190, 420, 405), "Recording Metadata", "metadata/odom_global.jsonl\nTimestamped base pose\nPosition + orientation", "#eff6ff", BLUE)
    box(d, (90, 515, 420, 730), "Calibration", "extrinsics.json\nbase_link -> front_lidar_link\nintrinsics.json for projection", "#fff7ed", YELLOW)
    box(d, (575, 320, 925, 565), "Build Poses", "T_map_lidar(t) = T_map_base(t) T_base_lidar\nInterpolate/lookup nearest valid odom sample per frame", "white", "#94a3b8")
    box(d, (1080, 320, 1430, 565), "Relative Motion", "T_current_lidar_from_previous_lidar = inverse(T_map_lidar(t)) T_map_lidar(t-1)", "#f0fdfa", CYAN)
    box(d, (1530, 320, 1745, 565), "Compensated Track", "Move predicted 3D track state into the current LiDAR frame before association.", "#ecfdf5", GREEN)
    arrow(d, (420, 300), (575, 405), BLUE)
    arrow(d, (420, 620), (575, 480), YELLOW)
    arrow(d, (925, 445), (1080, 445), CYAN)
    arrow(d, (1430, 445), (1530, 445), GREEN)
    d.text((160, 875), "Completed strict AGHRI-odom batch: odom compensation applied on 10048/10048 possible frame transitions.", font=F_BODY, fill=INK)
    save(img, "06_ego_motion_compensation.png")


def metrics_matrix() -> None:
    img, d = canvas()
    draw_title(d, "Eight-Combination AGHRI Tracking Evaluation", "Strict DeepFusionMOT-compatible output with AGHRI odometry compensation")
    headers = ["ID", "Camera", "LiDAR", "HOTA", "MOTA", "IDF1", "IDSW"]
    rows = [
        ("S1", "G YOLO", "G SECOND", "0.1739", "-0.0180", "0.1983", "35"),
        ("S2", "FT YOLO", "G SECOND", "0.1776", "-0.0437", "0.1966", "37"),
        ("S3", "G YOLO", "FT SECOND", "0.2911", "-0.0531", "0.3147", "61"),
        ("S4", "FT YOLO", "FT SECOND", "0.2886", "-0.0571", "0.3128", "60"),
        ("P1", "G YOLO", "G PointPillars", "0.1557", "-0.7200", "0.1458", "92"),
        ("P2", "FT YOLO", "G PointPillars", "0.1574", "-0.7314", "0.1462", "94"),
        ("P3", "G YOLO", "FT PointPillars", "0.3417", "-0.1878", "0.3424", "46"),
        ("P4", "FT YOLO", "FT PointPillars", "0.3420", "-0.1829", "0.3436", "42"),
    ]
    x = 90
    y = 185
    colw = [120, 260, 340, 150, 150, 150, 120]
    d.rounded_rectangle((80, 170, 1720, 870), radius=20, fill="white", outline="#cbd5e1", width=3)
    for i, h in enumerate(headers):
        d.text((x + 15, y), h, font=F_SUB, fill=INK)
        x += colw[i]
    y += 55
    for row in rows:
        fill = "#ecfdf5" if row[0] == "P4" else ("#f8fafc" if int(y / 55) % 2 == 0 else "white")
        d.rectangle((90, y - 8, 1690, y + 38), fill=fill)
        x = 90
        for i, val in enumerate(row):
            d.text((x + 15, y), val, font=F_BODY, fill=INK if row[0] == "P4" else DARK_GRAY)
            x += colw[i]
        y += 62
    d.text((115, 920), "Best strict+odom HOTA in this completed batch: P4, fine-tuned YOLO + fine-tuned PointPillars.", font=F_BODY, fill=GREEN)
    d.text((115, 960), "Metrics are AGHRI ZED RGB identity metrics using projected tracked 3D boxes and the bundled TrackEval metric family.", font=F_BODY, fill=MUTED)
    save(img, "07_offline_experiment_matrix_and_metrics.png")


def ros2_future() -> None:
    img, d = canvas()
    draw_title(d, "ROS 2 Integration Path", "Current ROS 2 fusion remains separate from the newly completed offline DeepFusionMOT tracking layer")
    box(d, (80, 200, 360, 410), "ROS Bag / Live Sensors", "ZED image topic\nLivox cloud topic\nTF and CameraInfo", "#eff6ff", BLUE)
    box(d, (480, 140, 790, 330), "Live/Cache YOLO", "Publishes camera detections for generic or fine-tuned YOLO.", "#eef2ff", BLUE)
    box(d, (480, 435, 790, 625), "Live/Cache LiDAR", "Publishes SECOND or PointPillars detections for generic or fine-tuned LiDAR model.", "#ecfdf5", GREEN)
    box(d, (910, 285, 1240, 500), "Current Online Fusion", "Synchronisation, projection, association, matched/unmatched/ALL3D outputs, diagnostics, RViz markers.", "#f0fdfa", CYAN)
    box(d, (1370, 285, 1690, 500), "Next ROS 2 Tracker Node", "Wrap DeepFusionMOTTracker.\nPublish tracked IDs, track markers, projected track images, diagnostics.", "#faf5ff", PURPLE)
    box(d, (520, 760, 1360, 925), "Current Status", "The full DeepFusionMOT tracker is implemented and evaluated offline for AGHRI. The ROS 2 system already runs detector publication, late fusion, and RViz. The remaining integration step is a ROS 2 node that keeps tracker state online.", "white", "#94a3b8")
    arrow(d, (360, 280), (480, 235), BLUE)
    arrow(d, (360, 340), (480, 530), GREEN)
    arrow(d, (790, 235), (910, 345), BLUE)
    arrow(d, (790, 530), (910, 430), GREEN)
    arrow(d, (1240, 392), (1370, 392), PURPLE)
    arrow(d, (1530, 500), (1180, 760), PURPLE)
    save(img, "08_ros2_future_tracking_extension.png")


def write_note() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = dedent(
        """\
        # AGHRI DeepFusionMOT Full Framework Note

        Generated by `tools/create_deepfusionmot_framework_note.py` on 2026-07-20.

        This note describes the complete DeepFusionMOT-style framework now present in the `late_fusion` repository. It is written for the AGHRI-first direction: merge the non-trainable DeepFusionMOT tracking logic into the existing AGHRI camera-LiDAR late-fusion framework, evaluate all AGHRI detector combinations, and then use the same tracker as the basis for a future ROS 2 tracking node.

        ## 1. Framework Overview

        ![AGHRI DeepFusionMOT framework overview](figures/01_framework_overview.png)

        DeepFusionMOT is a multi-object tracking framework, not a detector fine-tuning method. The original repository is KITTI-oriented and expects KITTI-style 2D detections, 3D detections, calibration, OXTS ego-motion files, and KITTI tracking evaluation. The useful part for AGHRI is the tracker design: detection-level fusion, 2D/3D Kalman state, staged association, track lifecycle management, and tracking metrics.

        In this repository, the AGHRI implementation keeps the existing late-fusion detection pipeline and adds a native DeepFusionMOT-style tracking layer after it. The tracker consumes typed `Detection2D`, `Detection3D`, and `FusionResult` objects directly instead of forcing AGHRI data through KITTI text files.

        ## 2. Original DeepFusionMOT vs AGHRI Adaptation

        | Aspect | Original DeepFusionMOT | AGHRI implementation in this repo |
        | --- | --- | --- |
        | Dataset shape | KITTI Tracking sequences | AGHRI recordings from `/media/prabuddhi/Backup2/Updated Dataset_PW` |
        | Camera detector | External 2D detections, commonly RRC in the original setup | YOLO11s generic or AGHRI fine-tuned YOLO caches/live node |
        | LiDAR detector | External 3D detections, commonly PointRCNN in the original setup | SECOND or PointPillars, generic or AGHRI fine-tuned |
        | Calibration | KITTI calib text files | `calibration/intrinsics.json` and `calibration/extrinsics.json` |
        | Ego motion | KITTI OXTS | AGHRI `metadata/odom_global.jsonl` plus `base_link -> front_lidar_link` calibration |
        | Tracker | Non-trainable geometric/Kalman tracker | Native port in `late_fusion_pkg/deepfusionmot_tracker.py` |
        | Metrics | KITTI/TrackEval tracking metrics | Bundled TrackEval metrics on AGHRI ZED RGB identity annotations and projected tracked 3D boxes |
        | ROS 2 | Not native to the original clone | Current repo has ROS 2 detection/fusion/RViz; ROS 2 tracking node is the next integration step |

        ## 3. Detection and Fusion Inputs

        ![Detector and fusion inputs](figures/02_detector_and_fusion_inputs.png)

        The detector layer produces two evidence streams:

        - Camera: YOLO 2D person boxes in ZED RGB image coordinates.
        - LiDAR: SECOND or PointPillars 3D person boxes in the front LiDAR frame.

        The existing AGHRI late-fusion code projects every LiDAR 3D box into the image using the verified camera-LiDAR calibration:

        ```text
        x_image ~ P_camera * T_camera_lidar * X_lidar
        ```

        The projected LiDAR footprint is matched to YOLO boxes by 2D IoU:

        ```text
        IoU = area(projected_3D_box intersection YOLO_box) /
              area(projected_3D_box union YOLO_box)
        ```

        This produces the same semantic outputs used elsewhere in the repo:

        - matched 3D detections: LiDAR boxes supported by a camera detection.
        - unmatched camera detections: YOLO boxes without a matched LiDAR box.
        - unmatched LiDAR detections: LiDAR boxes without a matched camera box.
        - ALL3D: matched LiDAR boxes plus unmatched LiDAR boxes, preserving detector geometry and scores.

        ## 4. Tracking Update Cycle

        ![Tracking update cycle](figures/03_tracking_update_cycle.png)

        The DeepFusionMOT tracker is stateful within each AGHRI recording. For each timestamped frame it:

        1. Loads or receives YOLO and LiDAR detections.
        2. Runs detection-level projection and late-fusion association.
        3. Predicts existing 2D and 3D tracks with Kalman filters.
        4. Optionally compensates 3D track states using AGHRI odometry.
        5. Associates detections to tracks using the DeepFusionMOT cascade.
        6. Updates matched tracks, creates new tentative tracks, and marks missed tracks.
        7. Outputs tracks according to the selected lifecycle/output policy.

        The strict AGHRI preset adopts the DeepFusionMOT Pedestrian settings that can be applied directly:

        | Setting | Strict value |
        | --- | ---: |
        | `min_hits` / `min_frames` | 3 |
        | `max_age_frames` / `max_ages` | 25 |
        | 3D association gate | 0.01 |
        | 2D association gate | 0.50 |
        | 3D IDs | even IDs |
        | 2D IDs | odd IDs |
        | lifecycle | frame-count DeepFusionMOT mode |
        | output mode | confirmed tracks plus early-frame tracks while `frame_count <= min_hits` |

        ## 5. Deep Association Cascade

        ![Deep association cascade](figures/04_deep_association_cascade.png)

        The AGHRI implementation follows the original DeepFusionMOT idea that stronger fused evidence should be consumed first:

        1. Fused/matched 3D detections are associated to active 3D tracks.
        2. LiDAR-only 3D detections are associated to remaining 3D tracks.
        3. Camera-only 2D detections are associated to remaining 2D tracks.
        4. 2D tracks can support projected 3D tracks.

        Association uses Hungarian assignment. The strict preset uses BEV IoU as the AGHRI 3D proxy because AGHRI boxes are in LiDAR coordinates and the current detector outputs are already evaluated through BEV/3D geometry. The 2D stages use image-space IoU.

        ## 6. Track Lifecycle and IDs

        ![Track lifecycle and ID semantics](figures/05_track_lifecycle.png)

        A new detection creates a tentative track. In strict mode, tentative tracks that fail to match are deleted immediately, while confirmed tracks can enter a reactivation/missed state until `max_age_frames=25`. A track is eligible for normal output after enough hits. This differs from the earlier detection-only framework because track IDs are persistent temporal identities, not per-frame detection labels.

        The even/odd ID policy is intentional:

        - even IDs identify 3D tracks.
        - odd IDs identify 2D tracks.

        This mirrors the original DeepFusionMOT convention and makes mixed 2D/3D tracking diagnostics easier to inspect.

        ## 7. AGHRI Ego-Motion Compensation

        ![AGHRI ego-motion compensation](figures/06_ego_motion_compensation.png)

        The original DeepFusionMOT code uses KITTI OXTS. The AGHRI port uses the odometry files inside each recording:

        ```text
        <recording>/metadata/odom_global.jsonl
        ```

        Together with calibration:

        ```text
        /media/prabuddhi/Backup2/Updated Dataset_PW/calibration/extrinsics.json
        /media/prabuddhi/Backup2/Updated Dataset_PW/calibration/intrinsics.json
        ```

        the tracker builds:

        ```text
        T_map_lidar(t) = T_map_base(t) T_base_lidar
        T_current_lidar_from_previous_lidar =
            inverse(T_map_lidar(t)) T_map_lidar(t-1)
        ```

        This relative motion moves the previous 3D track state into the current LiDAR frame before association. The completed strict AGHRI-odom batch reports odom compensation on `10048/10048` possible frame transitions.

        ## 8. Offline AGHRI Experiment Matrix and Metrics

        ![Offline experiment matrix and metrics](figures/07_offline_experiment_matrix_and_metrics.png)

        The completed strict AGHRI-odom evaluation covers all eight requested detector combinations:

        | ID | Combination | HOTA | MOTA | IDF1 | IDSW |
        | --- | --- | ---: | ---: | ---: | ---: |
        | S1 | G YOLO + G SECOND | 0.1739 | -0.0180 | 0.1983 | 35 |
        | S2 | FT YOLO + G SECOND | 0.1776 | -0.0437 | 0.1966 | 37 |
        | S3 | G YOLO + FT SECOND | 0.2911 | -0.0531 | 0.3147 | 61 |
        | S4 | FT YOLO + FT SECOND | 0.2886 | -0.0571 | 0.3128 | 60 |
        | P1 | G YOLO + G PointPillars | 0.1557 | -0.7200 | 0.1458 | 92 |
        | P2 | FT YOLO + G PointPillars | 0.1574 | -0.7314 | 0.1462 | 94 |
        | P3 | G YOLO + FT PointPillars | 0.3417 | -0.1878 | 0.3424 | 46 |
        | P4 | FT YOLO + FT PointPillars | 0.3420 | -0.1829 | 0.3436 | 42 |

        Full metrics:

        ```text
        results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_deepfusionmot_strict_aghri_odom_metrics/deepfusionmot_metrics.md
        ```

        The best strict+odom HOTA in the completed batch is P4: fine-tuned YOLO + fine-tuned PointPillars. The result should be interpreted as an AGHRI ZED RGB identity-tracking metric using projected tracked 3D boxes, not as official KITTI server performance.

        ## 9. ROS 2 Integration Path

        ![ROS 2 future tracking extension](figures/08_ros2_future_tracking_extension.png)

        The current ROS 2 pipeline already supports:

        ```text
        rosbag/live sensors
        -> YOLO detector publisher or cache publisher
        -> SECOND / PointPillars detector publisher or cache publisher
        -> AGHRI late-fusion node
        -> visualisation node
        -> RViz
        ```

        The DeepFusionMOT tracker is currently implemented and evaluated offline. The next ROS 2 step is to wrap `DeepFusionMOTTracker` in a ROS 2 node that subscribes to the existing fusion/detection topics, preserves tracker state online, and publishes tracked 3D IDs, tracked 2D IDs, marker arrays, projected tracking images, and diagnostics.

        ## 10. Reproduction Commands

        Run the strict AGHRI-odom eight-combination tracker batch:

        ```bash
        PYTHONPATH=. /usr/bin/python3 tools/run_aghri_deepfusionmot_batch.py \\
          --recording footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label \\
          --output-dir results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_deepfusionmot_strict_aghri_odom \\
          --limit-frames 0 \\
          --visual-frames 0 \\
          --ego-motion-mode aghri_odom \\
          --calibration-path /media/prabuddhi/Backup2/Updated Dataset_PW/calibration \\
          --deepfusionmot-strict
        ```

        Evaluate with the bundled TrackEval metric family:

        ```bash
        PYTHONPATH=. /usr/bin/python3 tools/evaluate_aghri_deepfusionmot_metrics.py \\
          --batch-root results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_deepfusionmot_strict_aghri_odom \\
          --output-dir results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_deepfusionmot_strict_aghri_odom_metrics
        ```

        Regenerate this framework note and all diagrams:

        ```bash
        PYTHONPATH=. /usr/bin/python3 tools/create_deepfusionmot_framework_note.py
        ```

        ## 11. Boundaries

        - The tracker/fusion layer is not a trainable neural network. The trainable components remain YOLO, SECOND, and PointPillars.
        - The AGHRI port intentionally avoids forcing AGHRI through the original KITTI file layout.
        - Detection-level ALL3D still preserves original LiDAR detector geometry; tracking adds temporal IDs and state, not detector box regression.
        - Current ROS 2 fusion/RViz is operational, but the full DeepFusionMOT tracker is currently offline. A ROS 2 tracking node is the next implementation stage.
        """
    )
    (OUT_DIR / "DEEPFUSIONMOT_FULL_FRAMEWORK_NOTE.md").write_text(text, encoding="utf-8")

    manifest = "\n".join(
        [
            "figure,path,description",
            "01,figures/01_framework_overview.png,End-to-end AGHRI DeepFusionMOT framework overview",
            "02,figures/02_detector_and_fusion_inputs.png,Detector and detection-level late-fusion inputs",
            "03,figures/03_tracking_update_cycle.png,Per-frame DeepFusionMOT tracking update cycle",
            "04,figures/04_deep_association_cascade.png,Staged DeepFusionMOT association cascade",
            "05,figures/05_track_lifecycle.png,Track lifecycle and ID semantics",
            "06,figures/06_ego_motion_compensation.png,AGHRI odometry compensation",
            "07,figures/07_offline_experiment_matrix_and_metrics.png,Eight-combination offline metrics summary",
            "08,figures/08_ros2_future_tracking_extension.png,ROS 2 tracking integration path",
        ]
    )
    (OUT_DIR / "visual_manifest.csv").write_text(manifest + "\n", encoding="utf-8")


def main() -> None:
    framework_overview()
    detector_fusion_inputs()
    tracking_update_cycle()
    association_cascade()
    lifecycle()
    ego_motion()
    metrics_matrix()
    ros2_future()
    write_note()
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
