"""ROS 2 generic KITTI-pretrained SECOND detector node for AGHRI Livox clouds."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def main(args=None) -> None:
    import sys
    sys.path.insert(0, "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
    from mmdet3d.apis import init_model, inference_detector
    import torch

    class AghriSecondDetector(Node):
        def __init__(self):
            super().__init__("aghri_second_detector")
            self.declare_parameter("points_topic", "/dataset/lidar/points")
            self.declare_parameter("detections_topic", "/detector/lidar/detections")
            self.declare_parameter("config", "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d/configs/models/second_kitti_pretrained_eval_aghri.py")
            self.declare_parameter("checkpoint", "/home/prabuddhi/Desktop/late_fusion/deepfusionmot/checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth")
            self.declare_parameter("device", "cuda:0")
            self.declare_parameter("score_threshold", 0.0)
            self.declare_parameter("pedestrian_class_id", 0)

            requested = str(self.get_parameter("device").value)
            device = requested if requested == "cpu" or torch.cuda.is_available() else "cpu"
            self.model = init_model(str(self.get_parameter("config").value), str(self.get_parameter("checkpoint").value), device=device)
            self.publisher = self.create_publisher(Detection3DArray, self.get_parameter("detections_topic").value, 10)
            self.subscription = self.create_subscription(PointCloud2, self.get_parameter("points_topic").value, self.callback, 10)

        def callback(self, msg: PointCloud2) -> None:
            xyz = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
            if xyz.size == 0:
                xyzi = np.empty((0, 4), dtype=np.float32)
            else:
                xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
                intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                xyzi = np.column_stack([xyz, intensity]).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as stream:
                temp_path = Path(stream.name)
                xyzi.tofile(stream)
            try:
                result = inference_detector(self.model, str(temp_path))
            finally:
                temp_path.unlink(missing_ok=True)
            if isinstance(result, (tuple, list)):
                result = result[0]
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()

            out = Detection3DArray()
            out.header = msg.header
            out.header.frame_id = msg.header.frame_id or "front_lidar_link"
            threshold = float(self.get_parameter("score_threshold").value)
            pedestrian_id = int(self.get_parameter("pedestrian_class_id").value)
            for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if int(label) != pedestrian_id or float(score) < threshold:
                    continue
                det = Detection3D()
                det.id = str(idx)
                bbox = BoundingBox3D()
                bbox.center.position.x = float(box[0])
                bbox.center.position.y = float(box[1])
                bbox.center.position.z = float(box[2] + box[5] / 2.0)
                bbox.center.orientation.w = 1.0
                bbox.size.x = float(box[3])
                bbox.size.y = float(box[4])
                bbox.size.z = float(box[5])
                det.bbox = bbox
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(label))
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                out.detections.append(det)
            self.publisher.publish(out)

    rclpy.init(args=args)
    node = AghriSecondDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
