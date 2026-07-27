import hashlib
import json
import pickle
from pathlib import Path

from scripts.verify_staged_aghri_second_dataset import verify_dataset
from tools.aghri_second_finetune_common import ensure_no_split_overlap, normalize_box
from tools.import_aghri_second_cluster_run import copy_with_sha, find_best_checkpoint


def test_normalize_box_uses_bottom_center_and_rz_yaw():
    box = normalize_box([1, 2, 0.3, 0.4, 0.5, 1.6, 0.0, 0.0, 1.57])
    assert box == [1.0, 2.0, 0.3, 0.4, 0.5, 1.6, 1.57]


def test_split_overlap_detection_reports_leakage():
    overlaps = ensure_no_split_overlap(
        {
            "train": ["scene_a", "scene_b"],
            "val": ["scene_c", "scene_b"],
            "test": ["scene_d"],
        }
    )
    assert overlaps == {"train_vs_val": ["scene_b"]}


def test_copy_with_sha_preserves_checkpoint_hash(tmp_path):
    source = tmp_path / "source.pth"
    source.write_bytes(b"checkpoint")
    dest = tmp_path / "dest" / "best.pth"
    result = copy_with_sha(source, dest)
    assert dest.exists()
    assert result["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_find_best_checkpoint_prefers_best_file(tmp_path):
    root = tmp_path / "cluster"
    latest = root / "outputs" / "run" / "epoch_1.pth"
    best = root / "outputs" / "run" / "best_aghri_bev_ap_mean_epoch_1.pth"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"latest")
    best.write_bytes(b"best")
    assert find_best_checkpoint(root) == best


def test_verify_staged_dataset_rejects_test_artifacts(tmp_path):
    data_root = tmp_path / "aghri_mmdet3d"
    points = data_root / "points" / "scene_a"
    infos = data_root / "infos"
    points.mkdir(parents=True)
    infos.mkdir(parents=True)
    (points / "000.bin").write_bytes(b"\x00" * 16)
    payload = {
        "metainfo": {"classes": ["person"]},
        "data_list": [
            {
                "sample_idx": "scene_a/000",
                "lidar_points": {"lidar_path": "points/scene_a/000.bin", "num_pts_feats": 4},
                "instances": [{"bbox_3d": [1, 2, 0, 0.5, 0.5, 1.5, 0], "bbox_label_3d": 0}],
            }
        ],
    }
    for split in ("train", "val", "test"):
        with (infos / f"agri_person_infos_{split}.pkl").open("wb") as stream:
            pickle.dump(payload, stream)
    summary = verify_dataset(data_root)
    assert not summary["ok"]
    assert "Held-out test split artifacts" in " ".join(summary["errors"])
