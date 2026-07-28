#!/usr/bin/env python3
"""Generate same-frame DeepFusionMOT continuation visuals for the walkthrough."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = ROOT / "results" / "late_fusion_visual_walkthrough"
FIGURES = WALKTHROUGH / "figures"
DATA_PATH = WALKTHROUGH / "late_fusion_worked_calculation.json"

CANVAS_W = 1600
CANVAS_H = 920
BG = (247, 249, 252)
INK = (20, 33, 61)
MUTED = (82, 96, 112)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
CYAN = (8, 145, 178)
RED = (220, 38, 38)
YELLOW = (202, 138, 4)
PURPLE = (124, 58, 237)
GRAY = (226, 232, 240)
DARK_GRAY = (55, 65, 81)
WHITE = (255, 255, 255)
PALE_GREEN = (220, 252, 231)
PALE_BLUE = (239, 246, 255)
PALE_RED = (254, 226, 226)
PALE_PURPLE = (250, 245, 255)
PALE_CYAN = (240, 253, 250)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation2")):
        path = base / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_SUB = font(23, True)
F_BODY = font(20)
F_SMALL = font(17)
F_TINY = font(15)


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def base_zed_image() -> Image.Image:
    """Recover the report frame from the saved Stage 1 figure."""

    # The existing figure is 1212 x 456: title band of 60 px, then the
    # original 672 x 376 image on the left.
    stage_1 = Image.open(FIGURES / "01_raw_zed_rgb_image.png").convert("RGB")
    return stage_1.crop((0, 60, 672, 436))


def scale_image(image: Image.Image, width: int) -> Image.Image:
    ratio = width / image.width
    return image.resize((width, int(image.height * ratio)), Image.Resampling.LANCZOS)


def title(draw: ImageDraw.ImageDraw, text: str, subtitle: str) -> None:
    draw.text((44, 34), text, font=F_TITLE, fill=INK)
    draw.text((46, 82), subtitle, font=F_BODY, fill=MUTED)


def rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], outline: tuple[int, int, int], width: int = 4) -> None:
    for off in range(width):
        draw.rectangle((xy[0] - off, xy[1] - off, xy[2] + off, xy[3] + off), outline=outline)


def line_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: tuple[int, int, int], width: int = 3) -> None:
    draw.line([(xy[0], xy[1]), (xy[2], xy[1]), (xy[2], xy[3]), (xy[0], xy[3]), (xy[0], xy[1])], fill=fill, width=width)


def dashed_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = xy
    dash = 9
    gap = 7
    for x in range(x1, x2, dash + gap):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=fill, width=3)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=fill, width=3)
    for y in range(y1, y2, dash + gap):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=fill, width=3)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=fill, width=3)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], width: int, spacing: int = 7) -> int:
    x, y = xy
    line_height = fnt.getbbox("Ag")[3] - fnt.getbbox("Ag")[1] + spacing
    for line in text.split("\n"):
        words = line.split()
        parts: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=fnt)
            if bbox[2] - bbox[0] <= width:
                current = trial
            else:
                if current:
                    parts.append(current)
                current = word
        if current:
            parts.append(current)
        for part in parts or [""]:
            draw.text((x, y), part, font=fnt, fill=fill)
            y += line_height
    return y


def panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    heading: str,
    body: str,
    outline: tuple[int, int, int] = GRAY,
    fill: tuple[int, int, int] = WHITE,
    heading_color: tuple[int, int, int] = INK,
) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 22, y1 + 18), heading, font=F_SUB, fill=heading_color)
    draw_wrapped(draw, (x1 + 22, y1 + 55), body, F_SMALL, DARK_GRAY, x2 - x1 - 44)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int] = MUTED, width: int = 4) -> None:
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        pts = [(ex, ey), (ex - 16 * sign, ey - 9), (ex - 16 * sign, ey + 9)]
    else:
        sign = 1 if dy >= 0 else -1
        pts = [(ex, ey), (ex - 9, ey - 16 * sign), (ex + 9, ey - 16 * sign)]
    draw.polygon(pts, fill=color)


def paste_frame(canvas: Image.Image, data: dict, xy: tuple[int, int], width: int, labels: bool = True) -> tuple[int, int, int, int]:
    frame = scale_image(base_zed_image(), width)
    canvas.paste(frame, xy)
    draw = ImageDraw.Draw(canvas)
    scale = frame.width / data["image_dimensions"][0]
    x0, y0 = xy

    yolo = tuple(int(round(v * scale)) for v in data["camera_detection"]["bbox_xyxy"])
    proj = tuple(int(round(v * scale)) for v in data["geometry"]["projected_lidar_rectangle_clipped"])
    inter = tuple(int(round(v * scale)) for v in data["association"]["intersection_xyxy"])

    yolo_xy = (x0 + yolo[0], y0 + yolo[1], x0 + yolo[2], y0 + yolo[3])
    proj_xy = (x0 + proj[0], y0 + proj[1], x0 + proj[2], y0 + proj[3])
    inter_xy = (x0 + inter[0], y0 + inter[1], x0 + inter[2], y0 + inter[3])

    rect(draw, yolo_xy, BLUE, 3)
    line_rect(draw, proj_xy, GREEN, 4)
    draw.rounded_rectangle(inter_xy, radius=3, outline=CYAN, width=3)
    draw.line([(inter_xy[0], inter_xy[1]), (inter_xy[2], inter_xy[3])], fill=CYAN, width=2)
    draw.line([(inter_xy[0], inter_xy[3]), (inter_xy[2], inter_xy[1])], fill=CYAN, width=2)
    if labels:
        draw.text((yolo_xy[0], max(y0, yolo_xy[1] - 24)), "YOLO 2D", font=F_SMALL, fill=BLUE)
        draw.text((proj_xy[0], max(y0, proj_xy[1] - 24)), "projected 3D", font=F_SMALL, fill=GREEN)
    return (x0, y0, x0 + frame.width, y0 + frame.height)


def save(canvas: Image.Image, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    canvas.save(FIGURES / name)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    return canvas, ImageDraw.Draw(canvas)


def figure_22(data: dict) -> None:
    canvas, draw = new_canvas()
    title(draw, "Stage 22 - Fusion Outputs Become Tracker Inputs", "Same report frame: one matched 3D detection, no unmatched detections")
    frame_xy = paste_frame(canvas, data, (55, 150), 760)
    draw.text((55, frame_xy[3] + 18), f"IoU = {data['association']['iou']:.3f}; decision = MATCHED", font=F_BODY, fill=GREEN)

    panel(draw, (900, 145, 1515, 290), "Matched 3D stream", "Active in this frame. The PointPillars 3D cuboid is camera-supported by YOLO and enters DeepFusionMOT association stage 1.", GREEN, PALE_GREEN, GREEN)
    panel(draw, (900, 325, 1515, 470), "Unmatched camera stream", "Empty in this frame. If present later, these detections maintain 2D-only tracks or support projected 3D tracks.", BLUE, PALE_BLUE, BLUE)
    panel(draw, (900, 505, 1515, 650), "Unmatched LiDAR stream", "Empty in this frame. If present later, these boxes are still preserved in ALL3D and can continue/create 3D tracks.", RED, PALE_RED, RED)
    panel(draw, (900, 710, 1515, 850), "Tracker receives", "matched_3D=1, unmatched_camera=0, unmatched_LiDAR=0, ALL3D=1", PURPLE, PALE_PURPLE, PURPLE)
    arrow(draw, (815, 330), (900, 220), GREEN)
    arrow(draw, (815, 360), (900, 398), BLUE)
    arrow(draw, (815, 390), (900, 578), RED)
    arrow(draw, (1208, 650), (1208, 710), PURPLE)
    save(canvas, "22_tracker_input_routing.png")


def figure_23(data: dict) -> None:
    canvas, draw = new_canvas()
    title(draw, "Stage 23 - Matched and Unmatched Routing Policy", "The inactive branches are shown because this real selected frame has no unmatched detections")
    paste_frame(canvas, data, (50, 150), 520)

    rows = [
        ("matched 3D", "count 1", "3D association stage 1: update an existing 3D track or create a new tentative 3D track.", GREEN, PALE_GREEN),
        ("unmatched LiDAR", "count 0", "3D association stage 2: preserve in ALL3D and try remaining 3D tracks; otherwise create tentative 3D.", RED, PALE_RED),
        ("unmatched camera", "count 0", "2D association stage 3: update/create 2D-only track; may later support a projected 3D track.", BLUE, PALE_BLUE),
        ("ALL3D", "count 1", "Complete 3D output before tracking: matched LiDAR plus unmatched LiDAR, with original detector geometry.", CYAN, PALE_CYAN),
    ]
    y = 160
    for name, count, desc, color, fill in rows:
        active = not count.endswith("0")
        draw.rounded_rectangle((640, y, 900, y + 85), radius=14, fill=fill if active else (241, 245, 249), outline=color if active else GRAY, width=3)
        draw.text((662, y + 18), name, font=F_SUB, fill=color if active else MUTED)
        draw.text((662, y + 50), count, font=F_BODY, fill=color if active else MUTED)
        arrow(draw, (900, y + 42), (960, y + 42), color if active else MUTED)
        panel(draw, (960, y - 5, 1530, y + 95), "Routing", desc, color if active else GRAY, WHITE, color if active else MUTED)
        y += 155

    draw.text((60, 735), "Important: this figure does not invent unmatched people. It shows the real counts for the selected frame and the policy used when unmatched detections occur in other frames.", font=F_SMALL, fill=DARK_GRAY)
    save(canvas, "23_matched_unmatched_routing_policy.png")


def figure_24(data: dict) -> None:
    canvas, draw = new_canvas()
    title(draw, "Stage 24 - DeepFusionMOT Association on This Frame", "The matched cuboid is the strongest evidence and is consumed first")
    paste_frame(canvas, data, (55, 150), 700)
    draw.text((80, 565), "matched 3D detection", font=F_SUB, fill=GREEN)
    draw.text((80, 600), "score = %.3f, fused from YOLO + PointPillars" % data["association"]["fused_score"], font=F_BODY, fill=DARK_GRAY)

    panel(draw, (850, 145, 1515, 255), "1. Predict existing tracks", "Kalman prediction advances prior 3D/2D states. AGHRI odometry can compensate 3D states before matching.", YELLOW, (254, 252, 232), YELLOW)
    panel(draw, (850, 300, 1515, 410), "2. Associate matched 3D", "This frame's matched cuboid enters the first association stage. Strict 3D gate: BEV IoU >= 0.01.", GREEN, PALE_GREEN, GREEN)
    panel(draw, (850, 455, 1515, 565), "3. Update or create track", "If a prior 3D track matches, update it. If none matches, initialize a tentative even-ID 3D track.", PURPLE, PALE_PURPLE, PURPLE)
    panel(draw, (850, 610, 1515, 760), "4. Remaining detections", "Unmatched camera and LiDAR streams are empty for this selected frame, so later association stages have no extra detections to consume.", GRAY, WHITE, MUTED)
    arrow(draw, (1185, 255), (1185, 300), YELLOW)
    arrow(draw, (1185, 410), (1185, 455), GREEN)
    arrow(draw, (1185, 565), (1185, 610), PURPLE)
    arrow(draw, (755, 342), (850, 355), GREEN)
    save(canvas, "24_deepfusionmot_association_on_frame.png")


def figure_25(data: dict) -> None:
    canvas, draw = new_canvas()
    title(draw, "Stage 25 - Track State After the Frame Update", "Standalone selected-frame visualisation: the matched 3D box seeds or updates a 3D track")
    frame_xy = paste_frame(canvas, data, (55, 150), 760, labels=False)
    draw.text((365, 255), "track #0", font=F_SUB, fill=PURPLE)
    draw.text((365, 286), "tentative / fused_3d", font=F_SMALL, fill=PURPLE)
    draw.line([(342, 270), (330, 182)], fill=PURPLE, width=3)

    panel(draw, (895, 150, 1515, 320), "3D track record", "track_id = 0 (even 3D ID)\nstate = tentative\nupdate_source = fused_3d\nhits = 1, age = 1", PURPLE, PALE_PURPLE, PURPLE)
    panel(draw, (895, 360, 1515, 500), "Strict lifecycle", "A track becomes confirmed after enough hits. DeepFusionMOT Pedestrian strict mode uses min_hits = 3 and max_age_frames = 25.", GREEN, PALE_GREEN, GREEN)
    panel(draw, (895, 540, 1515, 680), "Output rule", "Strict output reports confirmed 3D tracks plus early-frame tracks while frame_count <= min_hits.", CYAN, PALE_CYAN, CYAN)
    panel(draw, (895, 720, 1515, 860), "What did not happen", "No unmatched camera-only or LiDAR-only track was created from this selected frame because both unmatched counts are zero.", GRAY, WHITE, MUTED)
    draw_wrapped(
        draw,
        (55, frame_xy[3] + 18),
        "The track geometry is still the original PointPillars cuboid; tracking adds temporal identity and state.",
        F_BODY,
        DARK_GRAY,
        780,
    )
    save(canvas, "25_tracker_state_after_update.png")


def figure_26(data: dict) -> None:
    canvas, draw = new_canvas()
    title(draw, "Stage 26 - How the Track Continues to the Next Frame", "Conceptual continuation drawn on the same report frame geometry")
    paste_frame(canvas, data, (55, 150), 720, labels=False)
    scale = 720 / data["image_dimensions"][0]
    proj = [int(round(v * scale)) for v in data["geometry"]["projected_lidar_rectangle_clipped"]]
    shifted = (55 + proj[0] + 16, 150 + proj[1] - 6, 55 + proj[2] + 16, 150 + proj[3] - 6)
    dashed_rect(draw, shifted, PURPLE)
    draw.text((shifted[0], max(150, shifted[1] - 26)), "predicted track #0", font=F_SMALL, fill=PURPLE)

    panel(draw, (860, 145, 1515, 265), "Before next detections", "Kalman predicts the track state from the previous frame.", PURPLE, PALE_PURPLE, PURPLE)
    panel(draw, (860, 310, 1515, 430), "With AGHRI odom", "If enabled, odom_global.jsonl moves the previous LiDAR-frame state into the current LiDAR frame.", YELLOW, (254, 252, 232), YELLOW)
    panel(draw, (860, 475, 1515, 595), "When new detections arrive", "Matched 3D, LiDAR-only, and camera-only detections are associated to the predicted tracks in cascade order.", GREEN, PALE_GREEN, GREEN)
    panel(draw, (860, 640, 1515, 800), "If no match occurs", "Tentative tracks delete immediately. Confirmed tracks enter reactivate/missed state and can survive until max_age_frames = 25.", RED, PALE_RED, RED)
    arrow(draw, (1188, 265), (1188, 310), PURPLE)
    arrow(draw, (1188, 430), (1188, 475), YELLOW)
    arrow(draw, (1188, 595), (1188, 640), GREEN)
    save(canvas, "26_next_frame_tracking_memory.png")


def thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, WHITE)
    out.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return out


def figure_27(data: dict) -> None:
    canvas = Image.new("RGB", (1900, 1220), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 35), "Stage 27 - Complete Same-Frame Flow into DeepFusionMOT", font=F_TITLE, fill=INK)
    draw.text((52, 82), "The report frame is followed from raw sensors through fusion outputs and then into tracking state.", font=F_BODY, fill=MUTED)

    items = [
        ("1. ZED frame", "01_raw_zed_rgb_image.png"),
        ("2. YOLO 2D", "03_yolo_detection.png"),
        ("3. PointPillars 3D", "04_pointpillars_3d_detection.png"),
        ("4. Projection", "09_projected_3d_cuboid.png"),
        ("5. IoU match", "11_iou_intersection.png"),
        ("6. Fusion outputs", "22_tracker_input_routing.png"),
        ("7. Tracker association", "24_deepfusionmot_association_on_frame.png"),
        ("8. Track state", "25_tracker_state_after_update.png"),
    ]
    positions = [(60, 150), (510, 150), (960, 150), (1410, 150), (60, 650), (510, 650), (960, 650), (1410, 650)]
    for idx, ((label, filename), pos) in enumerate(zip(items, positions)):
        x, y = pos
        draw.rounded_rectangle((x - 10, y - 10, x + 390, y + 390), radius=16, fill=WHITE, outline=GRAY, width=3)
        draw.text((x, y), label, font=F_SUB, fill=INK)
        image = thumb(FIGURES / filename, (360, 300))
        canvas.paste(image, (x + 10, y + 55))
        if idx not in (3, 7):
            sx = x + 390
            sy = y + 190
            ex = positions[idx + 1][0] - 20
            ey = positions[idx + 1][1] + 190
            arrow(draw, (sx, sy), (ex, ey), MUTED, 4)
    arrow(draw, (1600, 540), (260, 650), MUTED, 4)
    draw.text((70, 1110), "Selected-frame result: matched_3D=1, unmatched_camera=0, unmatched_LiDAR=0, ALL3D=1, then DeepFusionMOT adds track identity/state.", font=F_BODY, fill=DARK_GRAY)
    save(canvas, "27_complete_same_frame_tracking_flow.png")


def update_manifest() -> None:
    manifest_path = WALKTHROUGH / "visualisation_manifest.csv"
    rows: list[dict[str, str]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    fieldnames = ["figure_number", "filename", "purpose", "input_files", "data_rows", "script_used"]
    existing = {row["filename"]: row for row in rows if "filename" in row}
    additions = [
        ("22", "figures/22_tracker_input_routing.png", "Same-frame fusion outputs routed into tracker inputs"),
        ("23", "figures/23_matched_unmatched_routing_policy.png", "Matched/unmatched branch policy for the selected frame"),
        ("24", "figures/24_deepfusionmot_association_on_frame.png", "DeepFusionMOT association cascade applied to the selected frame"),
        ("25", "figures/25_tracker_state_after_update.png", "Track state created or updated from the selected matched 3D detection"),
        ("26", "figures/26_next_frame_tracking_memory.png", "How the selected-frame track state continues to the next frame"),
        ("27", "figures/27_complete_same_frame_tracking_flow.png", "Complete same-frame flow from sensors to DeepFusionMOT track state"),
    ]
    for number, filename, purpose in additions:
        existing[filename] = {
            "figure_number": number,
            "filename": filename,
            "purpose": purpose,
            "input_files": "late_fusion_worked_calculation.json; existing walkthrough figures",
            "data_rows": "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/1731407186_709929023",
            "script_used": "tools/create_late_fusion_tracking_walkthrough_extension.py",
        }
    rows = sorted(existing.values(), key=lambda row: (row.get("figure_number", ""), row.get("filename", "")))
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_simple_html() -> None:
    """Refresh the walkthrough HTML from the Markdown without dataset imports."""

    md_path = WALKTHROUGH / "LATE_FUSION_VISUAL_WALKTHROUGH.md"
    html_path = WALKTHROUGH / "LATE_FUSION_VISUAL_WALKTHROUGH.html"
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:DejaVu Sans,Arial,sans-serif;max-width:1050px;margin:32px auto;color:#14213d;}",
        "img{max-width:1000px;border:1px solid #e5e7eb;}",
        "pre{background:#f4f4f4;padding:10px;white-space:pre-wrap;}",
        "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:4px;font-size:9pt;}",
        "p,li{font-size:16px;line-height:1.45;}",
        "</style></head><body>",
    ]
    in_code = False
    in_table = False
    for line in lines:
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("# "):
            if in_table:
                out.append("</table>")
                in_table = False
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_table:
                out.append("</table>")
                in_table = False
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            out.append(f"<figure><img src='{html.escape(src, quote=True)}'><figcaption>{html.escape(alt)}</figcaption></figure>")
        elif line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if set("".join(cells)) <= {"-", ":", " "}:
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        elif not line.strip():
            if in_table:
                out.append("</table>")
                in_table = False
            out.append("<br>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            out.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        out.append("</table>")
    out.append("</body></html>")
    html_path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    data = load_data()
    figure_22(data)
    figure_23(data)
    figure_24(data)
    figure_25(data)
    figure_26(data)
    figure_27(data)
    update_manifest()
    write_simple_html()
    print("Wrote tracking continuation figures 22-27")


if __name__ == "__main__":
    main()
