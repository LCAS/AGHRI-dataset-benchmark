"""Persistent MMDetection3D worker for online ROS LiDAR inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import time

import numpy as np

from late_fusion_pkg.checkpoints import sha256_file
from late_fusion_pkg.live_conversions import decode_points
from late_fusion_pkg.live_worker_protocol import ProtocolError, recv_json, send_json


def _add_known_mmdet3d_paths() -> None:
    candidates = [
        "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d",
        "/home/prabuddhi/cluster/aghri_second_finetune/code/mmdetection3d",
        "/home/prabuddhi/cluster/aghri_pointpillars_finetune/code/mmdetection3d",
    ]
    for candidate in reversed(candidates):
        path = Path(candidate)
        if path.exists():
            sys.path.insert(0, str(path))


def _extract_predictions(result, selected_class_id: int, selected_class_name: str, score_threshold: float) -> list[dict]:
    if isinstance(result, (list, tuple)):
        result = result[0]
    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    out = []
    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        label = int(label)
        score = float(score)
        if label != int(selected_class_id) or score < float(score_threshold):
            continue
        x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
        values = np.asarray([x, y, z, length, width, height, yaw, score], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            continue
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            continue
        out.append(
            {
                "id": f"live_lidar_{index}",
                "x": x,
                "y": y,
                "z": z,
                "length": length,
                "width": width,
                "height": height,
                "yaw": yaw,
                "score": score,
                "label_id": label,
                "label_name": selected_class_name,
            }
        )
    return out


def _serve(args: argparse.Namespace) -> int:
    _add_known_mmdet3d_paths()
    from mmdet3d.apis import inference_detector, init_model
    import torch

    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"MMDetection3D checkpoint does not exist: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"MMDetection3D config does not exist: {config}")
    actual_sha = sha256_file(checkpoint)
    if args.checkpoint_sha256 and args.checkpoint_sha256 != "unknown" and actual_sha != args.checkpoint_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch for {checkpoint}: expected {args.checkpoint_sha256}, got {actual_sha}")

    resolved_device = args.device
    if resolved_device != "cpu" and not torch.cuda.is_available():
        resolved_device = "cpu"
    load_start = time.perf_counter()
    model = init_model(str(config), str(checkpoint), device=resolved_device)
    load_sec = time.perf_counter() - load_start

    metadata = {
        "worker": "mmdet3d_online",
        "dataset_profile": args.dataset_profile,
        "detector_family": args.lidar_detector,
        "lidar_model": args.lidar_model,
        "config_path": str(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "selected_class_id": int(args.selected_class_id),
        "selected_class_name": args.selected_class_name,
        "score_threshold": float(args.score_threshold),
        "box_origin": args.box_origin,
        "device": resolved_device,
        "model_load_sec": load_sec,
    }
    print(json.dumps({"event": "worker_ready", **metadata}), flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, int(args.port)))
        server.listen(1)
        conn, addr = server.accept()
        print(json.dumps({"event": "client_connected", "addr": str(addr)}), flush=True)
        with conn:
            while True:
                try:
                    request = recv_json(conn)
                except ProtocolError as exc:
                    print(json.dumps({"event": "protocol_error", "error": str(exc)}), flush=True)
                    return 2
                command = request.get("command", "infer")
                if command == "shutdown":
                    send_json(conn, {"status": "ok", "metadata": metadata})
                    return 0
                if command == "ping":
                    send_json(conn, {"status": "ok", "metadata": metadata})
                    continue
                if command != "infer":
                    send_json(conn, {"status": "error", "error": f"unsupported command {command!r}", "metadata": metadata})
                    continue

                start = time.perf_counter()
                temp_path = None
                try:
                    points = decode_points(str(request["points_b64"]), int(request["point_count"]))
                    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as stream:
                        temp_path = Path(stream.name)
                        points.astype(np.float32, copy=False).tofile(stream)
                    inference_start = time.perf_counter()
                    result = inference_detector(model, str(temp_path))
                    inference_sec = time.perf_counter() - inference_start
                    detections = _extract_predictions(
                        result,
                        selected_class_id=int(args.selected_class_id),
                        selected_class_name=args.selected_class_name,
                        score_threshold=float(args.score_threshold),
                    )
                    send_json(
                        conn,
                        {
                            "status": "ok",
                            "detections": detections,
                            "timings": {
                                "inference_sec": inference_sec,
                                "worker_total_sec": time.perf_counter() - start,
                            },
                            "metadata": metadata,
                        },
                    )
                except Exception as exc:
                    send_json(conn, {"status": "error", "error": str(exc), "metadata": metadata})
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True)
    parser.add_argument("--dataset-profile", required=True)
    parser.add_argument("--lidar-detector", required=True)
    parser.add_argument("--lidar-model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selected-class-id", type=int, required=True)
    parser.add_argument("--selected-class-name", required=True)
    parser.add_argument("--box-origin", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
