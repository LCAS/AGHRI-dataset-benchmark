import numpy as np

from late_fusion_pkg.aghri_mot_eval import (
    bbox_iou_matrix,
    build_trackeval_sequence,
    evaluate_sequence,
)


def test_bbox_iou_matrix_uses_xyxy_boxes():
    iou = bbox_iou_matrix([(0.0, 0.0, 10.0, 10.0)], [(5.0, 5.0, 15.0, 15.0)])
    assert np.isclose(iou[0, 0], 25.0 / 175.0)


def test_trackeval_sequence_perfect_tracking_scores_as_perfect():
    manifest_rows = [
        {"image_path": "/tmp/000001.png"},
        {"image_path": "/tmp/000002.png"},
    ]
    annotations = {
        "000001.png": [{"Class": "01", "BoundingBoxes": [0.0, 0.0, 10.0, 10.0]}],
        "000002.png": [{"Class": "01", "BoundingBoxes": [1.0, 0.0, 10.0, 10.0]}],
    }
    tracked = {
        0: [{"track_id": 7, "bbox_xyxy": (0.0, 0.0, 10.0, 10.0), "state": "confirmed"}],
        1: [{"track_id": 7, "bbox_xyxy": (1.0, 0.0, 11.0, 10.0), "state": "confirmed"}],
    }
    data, metadata = build_trackeval_sequence(
        manifest_rows=manifest_rows,
        annotations_by_file=annotations,
        tracked_by_frame=tracked,
    )
    metrics = evaluate_sequence(data)
    assert metadata["num_gt_ids"] == 1
    assert metadata["num_tracker_ids"] == 1
    assert metrics["HOTA"] == 1.0
    assert metrics["DetA"] == 1.0
    assert metrics["AssA"] == 1.0
    assert metrics["MOTA"] == 1.0
    assert metrics["IDF1"] == 1.0
