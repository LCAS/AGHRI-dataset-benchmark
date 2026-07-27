from pathlib import Path
import csv
import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest

from late_fusion_pkg.aghri_manifest import TimestampedFile, nearest_neighbor
from late_fusion_pkg.aghri_calibration import load_aghri_camera_lidar_calibration
from late_fusion_pkg.aghri_detector_profiles import (
    CAMERA_CACHES,
    LIDAR_CACHES,
    resolve_detector_selection,
    selection_summary_lines,
)
from late_fusion_pkg.cache_replay import load_second_cache, load_yolo_cache
from late_fusion_pkg.checkpoints import KITTI_CLASS_MAPPING, PEDESTRIAN_CLASS_ID, YOLO_GENERIC_PATH, YOLO_GENERIC_SHA256, sha256_file
from late_fusion_pkg.detection_types import Detection2D, Detection3D
from late_fusion_pkg.fusion_core import fuse_detections, iou_2d
from late_fusion_pkg.metrics import box2d_ap
from late_fusion_pkg.projection import lidar_box_corners, project_lidar_box
from late_fusion_pkg.ros_adapters import detection3d_to_msg, detections3d_from_msg
from late_fusion_pkg.visualization_utils import (
    CUBOID_EDGES,
    VisualBoxSmoother,
    color_rgba,
    detections3d_to_marker_array,
    draw_detection2d_array,
    draw_projected_wireframes,
    project_box_wireframe,
    projected_vs_yolo_metrics,
    stable_marker_id,
    interpolate_yaw,
    wrap_angle,
)


P = np.array(
    [
        [100.0, 0.0, 50.0, 0.0],
        [0.0, 100.0, 50.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)
CAMERA_FROM_LIDAR = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _load_tool_module(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_yolo_hash_if_present():
    if YOLO_GENERIC_PATH.exists():
        assert sha256_file(YOLO_GENERIC_PATH) == YOLO_GENERIC_SHA256


def test_pedestrian_mapping_is_not_car_or_cyclist():
    assert KITTI_CLASS_MAPPING[PEDESTRIAN_CLASS_ID] == "Pedestrian"
    assert KITTI_CLASS_MAPPING[PEDESTRIAN_CLASS_ID] != "Car"
    assert KITTI_CLASS_MAPPING[PEDESTRIAN_CLASS_ID] != "Cyclist"


def test_aghri_detector_selector_defaults_to_finetuned_caches():
    selection = resolve_detector_selection(validate_paths=False)
    assert selection.detector_profile == "custom"
    assert selection.camera.model == "finetuned"
    assert selection.lidar.model == "finetuned"
    assert selection.camera.cache_csv == CAMERA_CACHES["finetuned"].cache_csv
    assert selection.lidar.detector_family == "second"
    assert selection.lidar.cache_csv == LIDAR_CACHES["second"]["finetuned"].cache_csv
    assert selection.lidar.box_origin == "center"


def test_aghri_detector_profile_sets_both_modalities():
    selection = resolve_detector_selection(detector_profile="generic_finetuned", validate_paths=False)
    assert selection.camera.model == "generic"
    assert selection.lidar.model == "finetuned"
    assert selection.camera.cache_csv == CAMERA_CACHES["generic"].cache_csv
    assert selection.lidar.detector_family == "second"
    assert selection.lidar.cache_csv == LIDAR_CACHES["second"]["finetuned"].cache_csv
    assert any("camera_model=generic" in line for line in selection_summary_lines(selection))
    assert any("lidar_model=finetuned" in line for line in selection_summary_lines(selection))
    assert any("lidar_detector=second" in line for line in selection_summary_lines(selection))
    assert any("cache_box_origin=center" in line for line in selection_summary_lines(selection))


def test_aghri_detector_selector_explicit_cache_override_wins(tmp_path):
    yolo_override = tmp_path / "detections.csv"
    yolo_override.write_text("sample_id,timestamp,empty_frame\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        '{"checkpoint_path": "manual.pt", "checkpoint_sha256": "manualhash"}',
        encoding="utf-8",
    )
    selection = resolve_detector_selection(
        detector_profile="generic_generic",
        yolo_cache_override=str(yolo_override),
        validate_paths=False,
    )
    assert selection.camera.model == "generic"
    assert selection.camera.cache_csv == yolo_override
    assert selection.camera.source == "explicit_override"
    assert selection.camera.checkpoint_path == "manual.pt"
    assert selection.camera.checkpoint_sha256 == "manualhash"
    assert selection.lidar.cache_csv == LIDAR_CACHES["second"]["generic"].cache_csv


def test_aghri_detector_selector_supports_generic_pointpillars():
    selection = resolve_detector_selection(
        camera_model="finetuned",
        lidar_detector="pointpillars",
        lidar_model="generic",
        validate_paths=False,
    )
    assert selection.camera.model == "finetuned"
    assert selection.lidar.model == "generic"
    assert selection.lidar.detector_family == "pointpillars"
    assert selection.lidar.cache_csv == LIDAR_CACHES["pointpillars"]["generic"].cache_csv
    assert selection.lidar.box_origin == "bottom_center"
    assert any("lidar_detector=pointpillars" in line for line in selection_summary_lines(selection))


def test_aghri_detector_selector_supports_finetuned_pointpillars():
    selection = resolve_detector_selection(
        camera_model="finetuned",
        lidar_detector="pointpillars",
        lidar_model="finetuned",
        validate_paths=False,
    )
    assert selection.lidar.model == "finetuned"
    assert selection.lidar.detector_family == "pointpillars"
    assert selection.lidar.cache_csv == LIDAR_CACHES["pointpillars"]["finetuned"].cache_csv
    assert selection.lidar.box_origin == "center"
    lines = selection_summary_lines(selection)
    assert "LiDAR detector family: PointPillars" in lines
    assert "LiDAR model: fine-tuned AGHRI PointPillars" in lines
    assert "Live detector inference: no" in lines


def test_aghri_detector_selector_rejects_unknown_lidar_detector():
    pytest = __import__("pytest")
    with pytest.raises(ValueError, match="lidar_detector"):
        resolve_detector_selection(lidar_detector="not_a_detector")


def test_aghri_calibration_loader_accepts_calibration_directory():
    pytest = __import__("pytest")
    calibration_dir = Path("/media/prabuddhi/Backup2/Updated Dataset_PW/calibration")
    if not calibration_dir.exists():
        pytest.skip(f"AGHRI calibration directory not mounted: {calibration_dir}")
    calibration = load_aghri_camera_lidar_calibration(calibration_dir)
    assert calibration["image_width"] == 672
    assert calibration["image_height"] == 376
    assert calibration["camera_p"].shape == (3, 4)
    assert calibration["camera_from_lidar"].shape == (4, 4)
    assert calibration["base_from_lidar"].shape == (4, 4)
    assert calibration["lidar_frame"] == "front_lidar_link"


def test_lidar_box_corner_generation_bottom_center():
    box = Detection3D(center_xyz=(10.0, 0.0, 0.0), size_lwh=(2.0, 1.0, 3.0), yaw=0.0, score=1.0, label_id=PEDESTRIAN_CLASS_ID)
    corners = lidar_box_corners(box)
    assert corners.shape == (8, 3)
    assert np.isclose(corners[:, 2].min(), 0.0)
    assert np.isclose(corners[:, 2].max(), 3.0)


def test_partial_projection_is_clipped_not_dropped():
    box = Detection3D(center_xyz=(4.0, 3.0, 0.0), size_lwh=(2.0, 2.0, 2.0), yaw=0.0, score=1.0, label_id=PEDESTRIAN_CLASS_ID)
    projected = project_lidar_box(box, P, CAMERA_FROM_LIDAR, image_width=100, image_height=100)
    assert projected.valid
    x1, y1, x2, y2 = projected.bbox_xyxy
    assert 0.0 <= x1 < x2 <= 99.0
    assert 0.0 <= y1 < y2 <= 99.0


def test_boxes_behind_camera_are_rejected():
    box = Detection3D(center_xyz=(-2.0, 0.0, 0.0), size_lwh=(1.0, 1.0, 1.0), yaw=0.0, score=1.0, label_id=PEDESTRIAN_CLASS_ID)
    projected = project_lidar_box(box, P, CAMERA_FROM_LIDAR, image_width=100, image_height=100)
    assert not projected.valid
    assert projected.reason == "behind_camera"


def test_iou_2d_float_area():
    assert iou_2d((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_fusion_matches_and_preserves_unmatched_lidar():
    cam = [Detection2D((30, 10, 70, 60), 0.8, 0, detection_id="cam0")]
    lidar = [
        Detection3D((5.0, 0.0, 0.0), (1.0, 1.0, 2.0), 0.0, 0.7, PEDESTRIAN_CLASS_ID, detection_id="lidar0"),
        Detection3D((5.0, -8.0, 0.0), (1.0, 1.0, 2.0), 0.0, 0.6, PEDESTRIAN_CLASS_ID, detection_id="lidar1"),
    ]
    result = fuse_detections(cam, lidar, P, CAMERA_FROM_LIDAR, 100, 100, iou_threshold=0.01)
    assert len(result.matches) == 1
    assert len(result.matched_3d) == 1
    assert len(result.unmatched_lidar) == 1
    assert len(result.final_3d) == 2
    assert len(result.final_3d) == len(result.matched_3d) + len(result.unmatched_lidar)
    assert result.output_policy == "matched_plus_unmatched_lidar"


def test_empty_modalities_are_preserved():
    result = fuse_detections([], [], P, CAMERA_FROM_LIDAR, 100, 100)
    assert result.matches == ()
    assert result.final_3d == ()
    result = fuse_detections([Detection2D((0, 0, 1, 1), 0.5, 0)], [], P, CAMERA_FROM_LIDAR, 100, 100)
    assert len(result.unmatched_camera) == 1


def test_nearest_neighbor_with_unmatched_recording():
    anchors = [
        TimestampedFile(1.0, Path("a.pcd"), "lidar"),
        TimestampedFile(2.0, Path("b.pcd"), "lidar"),
    ]
    candidates = [TimestampedFile(1.05, Path("a.png"), "camera")]
    pairs = nearest_neighbor(anchors, candidates, max_delta_sec=0.1)
    assert pairs[0][1] is not None
    assert pairs[1][1] is None


def test_average_precision_ranks_predictions():
    metrics = box2d_ap(
        [
            (0.9, (0, 0, 10, 10)),
            (0.1, (50, 50, 60, 60)),
        ],
        [(0, 0, 10, 10)],
        threshold=0.5,
    )
    assert metrics.ap == 1.0
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0


def test_cached_yolo_timestamp_lookup_and_empty_output(tmp_path):
    path = tmp_path / "yolo.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "timestamp", "image_path", "x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "rec_a/123",
                "timestamp": "10.0",
                "image_path": "10.png",
                "x1": "1",
                "y1": "2",
                "x2": "3",
                "y2": "4",
                "confidence": "0.9",
                "class_id": "0",
                "class_name": "person",
            }
        )
    cache = load_yolo_cache(path, timestamp_tolerance_sec=0.05)
    assert len(cache.lookup("rec_a", 10.02)) == 1
    assert cache.lookup("rec_a", 10.2) == ()


def test_cached_second_timestamp_lookup_and_frame_id(tmp_path):
    path = tmp_path / "second.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_id",
                "timestamp",
                "pointcloud_path",
                "x",
                "y",
                "z",
                "length",
                "width",
                "height",
                "yaw",
                "score",
                "raw_label_id",
                "raw_label_name",
                "selected_pedestrian",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "rec_b/456",
                "timestamp": "20.0",
                "pointcloud_path": "20.pcd",
                "x": "1",
                "y": "2",
                "z": "-0.5",
                "length": "0.8",
                "width": "0.6",
                "height": "1.7",
                "yaw": "0.25",
                "score": "0.7",
                "raw_label_id": "0",
                "raw_label_name": "Pedestrian",
                "selected_pedestrian": "True",
            }
        )
    cache = load_second_cache(path, timestamp_tolerance_sec=0.05)
    detections = cache.lookup("rec_b", 20.0)
    assert len(detections) == 1
    assert detections[0].frame_id == "front_lidar_link"
    assert detections[0].box_origin == "bottom_center"
    assert np.isclose(detections[0].center_xyz[2], -0.5)
    center_origin_cache = load_second_cache(path, timestamp_tolerance_sec=0.05, box_origin="center")
    center_origin_detections = center_origin_cache.lookup("rec_b", 20.0)
    assert np.isclose(center_origin_detections[0].center_xyz[2], -1.35)
    velodyne_cache = load_second_cache(path, timestamp_tolerance_sec=0.05, frame_id="velodyne")
    assert velodyne_cache.lookup("rec_b", 20.0)[0].frame_id == "velodyne"
    assert velodyne_cache.validation.frame_ids == ("velodyne",)


def test_offline_lidar_csv_readers_convert_center_origin(tmp_path):
    path = tmp_path / "lidar.csv"
    fields = [
        "sample_id",
        "timestamp",
        "pointcloud_path",
        "x",
        "y",
        "z",
        "length",
        "width",
        "height",
        "yaw",
        "score",
        "raw_label_id",
        "raw_label_name",
        "selected_pedestrian",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "rec/1",
                "timestamp": "1.0",
                "pointcloud_path": "1.pcd",
                "x": "4.0",
                "y": "5.0",
                "z": "1.0",
                "length": "0.8",
                "width": "0.6",
                "height": "2.0",
                "yaw": "0.0",
                "score": "0.9",
                "raw_label_id": "0",
                "raw_label_name": "Pedestrian",
                "selected_pedestrian": "True",
            }
        )

    fusion_tool = _load_tool_module("run_aghri_generic_late_fusion")
    eval_tool = _load_tool_module("evaluate_aghri_generic_outputs")

    fusion_det = fusion_tool._read_lidar(path, box_origin="center")["rec/1"][0]
    assert fusion_det.box_origin == "bottom_center"
    assert np.isclose(fusion_det.center_xyz[2], 0.0)

    eval_box = eval_tool._read_lidar(path, box_origin="center")["rec/1"][0][1]
    assert np.isclose(eval_box[2], 0.0)
    assert np.isclose(eval_tool._read_lidar(path)["rec/1"][0][1][2], 1.0)


class _FakeHyp:
    def __init__(self):
        self.hypothesis = SimpleNamespace(class_id="", score=0.0)


class _FakeBBox3D:
    def __init__(self):
        self.center = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        self.size = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _FakeDetection3DMsg:
    def __init__(self):
        self.id = ""
        self.bbox = None
        self.results = []


def test_detection3darray_conversion_preserves_yaw_and_timestamp():
    det = Detection3D((1.0, 2.0, -0.5), (0.8, 0.6, 1.7), 0.5, 0.9, 0, detection_id="d0")
    msg_det = detection3d_to_msg(det, _FakeDetection3DMsg, _FakeBBox3D, _FakeHyp)
    assert np.isclose(msg_det.bbox.center.position.z, 0.35)
    msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=250000000), frame_id="front_lidar_link"),
        detections=[msg_det],
    )
    decoded = detections3d_from_msg(msg)
    assert len(decoded) == 1
    assert decoded[0].timestamp == 12.25
    assert decoded[0].frame_id == "front_lidar_link"
    assert decoded[0].box_origin == "bottom_center"
    assert np.isclose(decoded[0].center_xyz[2], -0.5)
    assert np.isclose(decoded[0].yaw, 0.5)


def test_humble_pathsubstitution_workaround_launch_file_is_static():
    launch_file = Path("launch/aghri_generic_late_fusion.launch.py")
    text = launch_file.read_text(encoding="utf-8")
    path_substitution_available = False
    if importlib.util.find_spec("launch") is not None:
        try:
            from launch.substitutions import PathSubstitution  # noqa: F401

            path_substitution_available = True
        except Exception:
            path_substitution_available = False
    assert path_substitution_available or "from launch_ros" not in text
    assert path_substitution_available or "import launch_ros" not in text
    for name in (
        "bag_path",
        "recording_name",
        "camera_model",
        "lidar_model",
        "lidar_detector",
        "detector_profile",
        "YOLO_cache",
        "SECOND_cache",
        "fusion_config",
        "launch_rviz",
        "enable_visualisation",
        "use_sim_time",
        "playback_rate",
    ):
        assert f'DeclareLaunchArgument("{name}"' in text
    for executable in (
        "cached_yolo_publisher",
        "cached_second_publisher",
        "aghri_late_fusion_node",
        "cache_diagnostics_collector",
        "aghri_late_fusion_visualizer",
    ):
        assert executable in text


def test_visualization_marker_ids_are_stable():
    assert stable_marker_id(0) == 0
    assert stable_marker_id(0, text=True) == 1
    assert stable_marker_id(12) == 24
    assert len(CUBOID_EDGES) == 12


def test_detection3d_marker_array_wireframe_and_deleteall():
    pytest = __import__("pytest")
    Marker = pytest.importorskip("visualization_msgs.msg").Marker
    MarkerArray = pytest.importorskip("visualization_msgs.msg").MarkerArray
    Point = pytest.importorskip("geometry_msgs.msg").Point
    Header = pytest.importorskip("std_msgs.msg").Header
    header = Header()
    header.stamp.sec = 1
    header.stamp.nanosec = 2
    header.frame_id = "front_lidar_link"
    det = Detection3D((1.0, 2.0, 0.0), (2.0, 1.0, 1.7), 0.25, 0.8, 0)
    arr = detections3d_to_marker_array(
        [det],
        header,
        MarkerArray,
        Marker,
        Point,
        box_namespace="matched_boxes",
        label_namespace="matched_labels",
        rgba=(0.0, 1.0, 0.0, 1.0),
        draw_labels=True,
        label_prefix="matched",
    )
    assert arr.markers[0].action == Marker.DELETEALL
    assert arr.markers[1].type == Marker.LINE_LIST
    assert arr.markers[1].ns == "matched_boxes"
    assert arr.markers[1].id == 0
    assert len(arr.markers[1].points) == 24
    assert arr.markers[2].type == Marker.TEXT_VIEW_FACING
    assert arr.markers[2].id == 1
    assert arr.markers[2].text == "matched#1:0.80"
    tracked = detections3d_to_marker_array(
        [Detection3D((1.0, 2.0, 0.0), (2.0, 1.0, 1.7), 0.25, 0.8, 0, detection_id="7")],
        header,
        MarkerArray,
        Marker,
        Point,
        box_namespace="tracked_boxes",
        label_namespace="tracked_labels",
        rgba=color_rgba("magenta"),
        draw_labels=True,
        label_prefix="track",
    )
    assert tracked.markers[2].text == "track#7:0.80"
    assert arr.markers[1].lifetime.sec == 0
    assert arr.markers[1].lifetime.nanosec == 250000000
    red = detections3d_to_marker_array(
        [det],
        header,
        MarkerArray,
        Marker,
        Point,
        box_namespace="unmatched_lidar_boxes",
        label_namespace="unmatched_lidar_labels",
        rgba=color_rgba("red"),
        draw_labels=True,
    )
    assert red.markers[1].ns == "unmatched_lidar_boxes"
    assert np.isclose(red.markers[1].color.r, 1.0)
    assert np.isclose(red.markers[1].color.g, 0.05)
    assert np.isclose(red.markers[1].color.b, 0.05)
    empty = detections3d_to_marker_array([], header, MarkerArray, Marker, Point, box_namespace="matched_boxes", label_namespace="matched_labels", rgba=(0, 1, 0, 1))
    assert len(empty.markers) == 1
    assert empty.markers[0].action == Marker.DELETEALL


def test_all3d_metric_audit_passes_and_counts_are_explicit():
    audit = _load_tool_module("audit_aghri_all3d_metrics")
    missing = [
        str(path)
        for combo in audit.COMBOS
        for path in (combo["metrics"], combo["overall"], combo["per_frame"])
        if not path.exists()
    ]
    if missing:
        pytest.skip("archived detector-comparison cache outputs are not in the cleaned repo")
    equality_rows, summary_rows, details = audit.build_rows()
    assert details["all_passed"] is True
    assert len(equality_rows) == 8
    assert len(summary_rows) == 8
    for row in equality_rows:
        assert row["Status"] == "PASS"
        assert abs(float(row["Difference"])) <= audit.TOLERANCE
        assert int(row["ALL3D prediction count"]) > 0
    for item in details["combinations"]:
        assert item["final_all3d_count"] == item["matched_pairs"] + item["unmatched_lidar_count"]
        assert item["final_all3d_count"] == item["lidar_prediction_count"]


def test_projected_wireframe_and_drawing_paths():
    cv2 = __import__("pytest").importorskip("cv2")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    det2d = Detection2D((10, 10, 30, 40), 0.9, 0, "person")
    drawn = draw_detection2d_array(image, [det2d], (255, 0, 0))
    assert int(drawn.sum()) > 0
    box = Detection3D((5.0, 0.0, 0.0), (1.0, 1.0, 2.0), 0.0, 0.7, PEDESTRIAN_CLASS_ID)
    lines, reason = project_box_wireframe(box, P, CAMERA_FROM_LIDAR, 100, 100)
    assert reason == "ok"
    assert lines
    projected, stats = draw_projected_wireframes(image, [box], P, CAMERA_FROM_LIDAR, (0, 255, 0))
    assert stats["drawn"] == 1
    assert int(projected.sum()) > 0


def test_projected_wireframe_rejects_boxes_behind_camera():
    box = Detection3D((-2.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0, 1.0, PEDESTRIAN_CLASS_ID)
    lines, reason = project_box_wireframe(box, P, CAMERA_FROM_LIDAR, 100, 100)
    assert lines == []
    assert reason == "behind_camera"


def test_projected_wireframe_clips_partial_and_rejects_fully_outside():
    partial = Detection3D((4.0, 3.0, 0.0), (2.0, 2.0, 2.0), 0.0, 1.0, PEDESTRIAN_CLASS_ID)
    lines, reason = project_box_wireframe(partial, P, CAMERA_FROM_LIDAR, 100, 100)
    assert reason == "ok"
    assert lines
    assert all(0 <= value <= 99 for line in lines for point in line for value in point)
    outside = Detection3D((5.0, 20.0, 0.0), (1.0, 1.0, 2.0), 0.0, 1.0, PEDESTRIAN_CLASS_ID)
    lines, reason = project_box_wireframe(outside, P, CAMERA_FROM_LIDAR, 100, 100)
    assert lines == []
    assert reason == "outside_image"


def test_projection_reason_accounting_includes_invalid_geometry():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid = Detection3D((5.0, 0.0, 0.0), (-1.0, 1.0, 2.0), 0.0, 1.0, PEDESTRIAN_CLASS_ID)
    _drawn, stats = draw_projected_wireframes(image, [invalid], P, CAMERA_FROM_LIDAR, (0, 255, 0))
    assert stats["drawn"] == 0
    assert stats["invalid_geometry"] == 1


def test_projected_vs_yolo_metrics_match_known_projection():
    lidar = Detection3D((5.0, 0.0, 0.0), (1.0, 1.0, 2.0), 0.0, 0.8, PEDESTRIAN_CLASS_ID)
    projected = project_lidar_box(lidar, P, CAMERA_FROM_LIDAR, 100, 100)
    camera = Detection2D(projected.bbox_xyxy, 0.9, 0, "person")
    metrics = projected_vs_yolo_metrics(lidar, camera, P, CAMERA_FROM_LIDAR, 100, 100)
    assert metrics["status"] == "ok"
    assert np.isclose(metrics["iou"], 1.0)
    assert np.isclose(metrics["centre_offset_px"], 0.0)
    assert np.isclose(metrics["height_ratio"], 1.0)
    assert np.isclose(metrics["width_ratio"], 1.0)


def test_visual_smoothing_handles_yaw_wraparound_without_mutating_input():
    previous = Detection3D((0.0, 0.0, 0.0), (1.0, 1.0, 2.0), math.pi - 0.1, 0.8, 0)
    current = Detection3D((1.0, 0.0, 0.0), (2.0, 1.0, 2.0), -math.pi + 0.1, 0.7, 0)
    smoother = VisualBoxSmoother(alpha=0.5, max_gap_sec=0.2, max_distance_m=1.5)
    assert smoother.update([previous], 1.0)[0] is previous
    output = smoother.update([current], 1.1)[0]
    assert np.isclose(output.center_xyz[0], 0.5)
    assert abs(abs(output.yaw) - math.pi) < 1e-6
    assert current.center_xyz == (1.0, 0.0, 0.0)
    assert np.isclose(interpolate_yaw(math.pi - 0.1, -math.pi + 0.1, 0.5), -math.pi)
    assert -math.pi <= wrap_angle(9.0) < math.pi


def test_source_time_tf_lookup_and_visual_defaults_are_static():
    node_text = Path("late_fusion_pkg/aghri_visualization_node.py").read_text(encoding="utf-8")
    assert "rclpy.time.Time.from_msg(source_stamp)" in node_text
    assert "camera_from_lidar(detections_msg.header.stamp)" in node_text
    assert "rclpy.time.Time()," not in node_text
    config_text = Path("config/aghri_visualization.yaml").read_text(encoding="utf-8")
    launch_text = Path("launch/aghri_generic_late_fusion.launch.py").read_text(encoding="utf-8")
    assert "publish_unmatched_lidar_markers: true" in config_text
    assert 'DeclareLaunchArgument("show_unmatched_lidar_markers", default_value="true")' in launch_text
    assert "publish_unmatched_lidar_markers:={show_unmatched_lidar_markers}" in launch_text
    assert 'DeclareLaunchArgument("lidar_detector", default_value="second")' in launch_text
    assert "enable_visual_smoothing: false" in config_text


def test_rviz_configs_parse_and_use_uncluttered_orbit_defaults():
    yaml = __import__("pytest").importorskip("yaml")
    for path in (
        Path("rviz/aghri_generic_late_fusion.rviz"),
        Path("rviz/aghri_generic_late_fusion_3d_only.rviz"),
    ):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        manager = config["Visualization Manager"]
        assert manager["Global Options"]["Fixed Frame"] == "front_lidar_link"
        assert manager["Views"]["Current"]["Class"] == "rviz_default_plugins/Orbit"
        displays = {display["Name"]: display for display in manager["Displays"]}
        unmatched_name = next(name for name in displays if name.startswith("Unmatched LiDAR Boxes"))
        original_name = next(name for name in displays if name.startswith("Original LiDAR Boxes"))
        assert displays[original_name]["Enabled"] is True
        assert displays[unmatched_name]["Enabled"] is True
        assert displays["TF"]["Enabled"] is False
        if path.name == "aghri_generic_late_fusion.rviz":
            assert displays["Tracked DeepFusionMOT Boxes"]["Topic"]["Value"] == "/visualization/tracked_3d_markers"
            assert displays["Tracked DeepFusionMOT Projection Image"]["Topic"]["Value"] == "/visualization/tracked_projection_image"
