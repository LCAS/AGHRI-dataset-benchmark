"""Focused tests for the canonical AGHRI ZED RGB fusion workflow."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import convert_aghri_test_to_rosbags as convert
import run_aghri_generic_vs_finetuned as compare
import verify_aghri_test_rosbags as verify


def _write_recording(root: Path, name: str) -> Path:
    rec = root / "dataset_part1" / name
    (rec / "annotations").mkdir(parents=True)
    (rec / "sensor_data/cam_zed_rgb").mkdir(parents=True)
    (rec / "sensor_data/lidar").mkdir(parents=True)
    image = "1000_000000010.png"
    lidar = "1000_000000000.pcd"
    (rec / "sensor_data/cam_zed_rgb" / image).write_bytes(b"png")
    (rec / "sensor_data/lidar" / lidar).write_bytes(b"pcd")
    (rec / "annotations/cam_zed_rgb_ann.json").write_text(json.dumps([
        {"Timestamp": 1000.000000010, "File": image, "Labels": [{"Class": "01", "BoundingBoxes": [0, 0, 10, 10]}]},
    ]), encoding="utf-8")
    (rec / "annotations/lidar_ann.json").write_text(json.dumps([
        {"Timestamp": 1000.0, "File": lidar, "Labels": [{"Class": "01", "BoundingBoxes": [0, 0, 1, 1, 1, 1, 0, 0, 0]}]},
    ]), encoding="utf-8")
    return rec


def _write_manifest(path: Path, rec: Path, name: str, leakage: bool = False) -> None:
    fields = [
        "split",
        "recording_name",
        "recording_path",
        "supported",
        "train_val_leakage",
        "expected_synchronised_frame_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for idx in range(6):
            writer.writerow({
                "split": "test",
                "recording_name": f"{name}_{idx}",
                "recording_path": str(rec),
                "supported": "True",
                "train_val_leakage": "True" if leakage and idx == 0 else "False",
                "expected_synchronised_frame_count": "1",
            })


def test_checkpoint_hash_helper(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert compare.sha256_file(checkpoint) == (
        "47320987f9a49d5b00119b960f247a95"
        "6773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_two_checkpoint_result_schema():
    summary = {
        "overall": {
            "total_yolo_person_detections": 2,
            "matched_prediction_count": 1,
            "valid_association_rate": 0.5,
            "mean_lidar_center_error": 0.4,
            "median_lidar_center_error": 0.3,
            "p95_lidar_center_error": 0.8,
            "lidar_error_gt_1_0m_rate": 0.0,
            "mean_association_time_sec": 0.001,
        },
        "frame_overall": {"approx_fps": 10.0, "total_gt3d_instances": 3},
        "frame_count": 4,
    }
    row = compare.summary_row("Generic YOLO11s", summary, Path("/tmp/yolo11s.pt"), "abc")
    assert list(row) == [
        "model",
        "checkpoint_path",
        "checkpoint_sha256",
        "detections",
        "matched_valid_results",
        "valid_fusion_rate",
        "mean_3d_error_m",
        "median_3d_error_m",
        "p95_3d_error_m",
        "error_rate_above_1m",
        "detector_runtime_sec",
        "fusion_runtime_sec",
        "fps",
        "processed_frames",
        "gt_instances",
    ]


def test_manifest_grouping_and_no_train_validation_leakage(tmp_path):
    rec = _write_recording(tmp_path, "test_recording_label")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rec, "test_recording_label")
    recordings = convert.load_manifest_recordings(manifest, tmp_path)
    assert len(recordings) == 6
    assert all(row["expected_synchronised_frame_count"] == 1 for row in recordings.values())

    leaking = tmp_path / "leaking.csv"
    _write_manifest(leaking, rec, "test_recording_label", leakage=True)
    with pytest.raises(RuntimeError, match="leakage"):
        convert.load_manifest_recordings(leaking, tmp_path)


def test_dataset_tools_converter_command_is_thin_wrapper(tmp_path):
    rec = _write_recording(tmp_path, "test_recording_label")
    tools_root = tmp_path / "agri-human-dataset-tools"
    script = tools_root / "ros2bag/check_and_make_rosbag2.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder\n", encoding="utf-8")
    out = tmp_path / "bag"
    command = convert.converter_command(Path("/usr/bin/python3"), tools_root, rec, out)
    assert command == [
        "/usr/bin/python3",
        str(script),
        "--bag-dir",
        str(rec),
        "--make-rosbag",
        "--rosbag-out",
        str(out),
        "--write-tf",
        "--tf-parent",
        convert.TF_PARENT_FRAME,
        "--tf-child",
        convert.TF_CHILD_FRAME,
        "--tf-xyzrpy",
        "0,0,0,0,0,0",
        "--write-calibration",
        "--calibration-path",
        str(convert.DEFAULT_CALIBRATION_PATH),
        "--intrinsics-member",
        convert.DEFAULT_INTRINSICS_MEMBER,
        "--extrinsics-member",
        convert.DEFAULT_EXTRINSICS_MEMBER,
        "--camera-name",
        "cam_zed_rgb",
        "--camera-info-topic",
        convert.CAMERA_INFO_TOPIC,
        "--camera-info-cameras",
        ",".join(convert.CAMERA_INFO_TOPICS),
    ]


def test_bag_topic_names_frame_ids_and_timestamp_conversion():
    assert convert.IMAGE_TOPIC == "/dataset/cam_zed_rgb/image"
    assert convert.CAMERA_INFO_TOPIC == "/dataset/cam_zed_rgb/camera_info"
    assert convert.FISH_FRONT_CAMERA_INFO_TOPIC == "/dataset/cam_fish_front/camera_info"
    assert convert.FISH_LEFT_CAMERA_INFO_TOPIC == "/dataset/cam_fish_left/camera_info"
    assert convert.FISH_RIGHT_CAMERA_INFO_TOPIC == "/dataset/cam_fish_right/camera_info"
    assert convert.LIDAR_TOPIC == "/dataset/lidar/points"
    assert convert.TF_TOPIC == "/tf"
    assert convert.TF_STATIC_TOPIC == "/tf_static"
    assert convert.TF_PARENT_FRAME == "map"
    assert convert.TF_CHILD_FRAME == "base_link"
    assert convert.DATASET_TOOLS_IMAGE_FRAME == "front_left_camera_optical_frame"
    assert convert.DATASET_TOOLS_LIDAR_FRAME == "front_lidar_link"
    assert convert.CALIBRATION_CAMERA_FRAME == "front_left_camera_optical_frame"
    assert convert.CALIBRATION_LIDAR_FRAME == "front_lidar_link"
    assert verify.IMAGE_TOPIC == convert.IMAGE_TOPIC
    assert verify.LIDAR_TOPIC == convert.LIDAR_TOPIC


def test_rviz_and_launch_files_exist_with_expected_defaults():
    rviz = REPO_ROOT / "rviz/aghri_zed_livox_fusion.rviz"
    launch = REPO_ROOT / "launch/aghri_test_bag_fusion_rviz.launch.py"
    assert rviz.exists()
    assert launch.exists()
    launch_text = launch.read_text(encoding="utf-8")
    assert "PathSubstitution" not in launch_text
    assert "launch_ros" not in launch_text
    assert "bag_path" in launch_text
    assert "params_file" in launch_text
    assert "use_sim_time" in launch_text
    assert "launch_rviz" in launch_text
    assert "ExecuteProcess" in launch_text
    assert "aghri_zed_livox_yolo11s_finetuned.yaml" in launch_text
    rviz_text = rviz.read_text(encoding="utf-8")
    assert "/camera_lidar_fusion/result" in rviz_text
    assert "/dataset/cam_zed_rgb/image" in rviz_text
    assert "Fixed Frame: front_lidar_link" in rviz_text


def test_launch_description_generation_has_required_arguments():
    launch_path = REPO_ROOT / "launch/aghri_test_bag_fusion_rviz.launch.py"
    spec = importlib.util.spec_from_file_location("aghri_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda _name: str(REPO_ROOT)
    module.get_package_prefix = lambda _name: "/tmp/camera_lidar_fusion_install"
    description = module.generate_launch_description()
    entities = list(description.entities)
    names = {
        entity.name
        for entity in entities
        if entity.__class__.__name__ == "DeclareLaunchArgument"
    }
    assert {
        "bag_path",
        "params_file",
        "rviz_config",
        "playback_rate",
        "qos_overrides",
        "loop",
        "start_offset",
        "use_sim_time",
        "autoplay",
        "launch_rviz",
    } <= names
    text = launch_path.read_text(encoding="utf-8")
    assert "--params-file" in text
    assert "bag_play_base" in text
    assert "--loop" in text
    assert "--qos-profile-overrides-path" in text


def test_readme_documents_workflow_and_avoids_experimental_terms():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Camera-LiDAR Human Detection Fusion Workflow" in readme
    assert "Generic YOLO11s" in readme
    assert "AGHRI-fine-tuned YOLO11s" in readme
    assert "ROS 2 Test-Bag Conversion" in readme
    assert "<dataset_tools_repo>/ros2bag/check_and_make_rosbag2.py" in readme
    assert "/dataset/cam_zed_rgb/camera_info" in readme
    assert "ros_camera_info_tf" in readme
    assert "tools/convert_aghri_test_to_rosbags.py" in readme
    assert "--all-recordings" in readme
    assert "RViz2 Visualisation" in readme
    assert "hybrid" not in readme.lower()
    assert "2x2" not in readme.lower()
    assert "2 × 2" not in readme
