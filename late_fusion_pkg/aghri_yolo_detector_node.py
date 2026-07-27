"""ROS 2 generic YOLO11s camera detector node for AGHRI ZED RGB."""

from __future__ import annotations


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
    from cv_bridge import CvBridge
    from ultralytics import YOLO
    import torch

    class AghriYoloDetector(Node):
        def __init__(self):
            super().__init__("aghri_yolo11s_detector")
            self.declare_parameter("image_topic", "/dataset/cam_zed_rgb/image")
            self.declare_parameter("detections_topic", "/detector/camera/detections")
            self.declare_parameter("checkpoint", "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/checkpoints/generic/yolo11s.pt")
            self.declare_parameter("confidence_threshold", 0.10)
            self.declare_parameter("imgsz", 640)
            self.declare_parameter("classes", [0])
            self.declare_parameter("device", "auto")

            device_param = self.get_parameter("device").value
            self.device = "cuda" if device_param == "auto" and torch.cuda.is_available() else ("cpu" if device_param == "auto" else device_param)
            self.model = YOLO(str(self.get_parameter("checkpoint").value)).to(self.device)
            self.bridge = CvBridge()
            self.publisher = self.create_publisher(Detection2DArray, self.get_parameter("detections_topic").value, 10)
            self.subscription = self.create_subscription(Image, self.get_parameter("image_topic").value, self.callback, 10)

        def callback(self, msg: Image) -> None:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            result = self.model(
                image,
                imgsz=int(self.get_parameter("imgsz").value),
                conf=float(self.get_parameter("confidence_threshold").value),
                classes=list(self.get_parameter("classes").value),
                verbose=False,
            )[0]
            out = Detection2DArray()
            out.header = msg.header
            boxes = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2, score, cls_id = box[:6]
                det = Detection2D()
                det.id = str(idx)
                bbox = BoundingBox2D()
                bbox.center.position.x = float((x1 + x2) / 2.0)
                bbox.center.position.y = float((y1 + y2) / 2.0)
                bbox.center.theta = 0.0
                bbox.size_x = float(x2 - x1)
                bbox.size_y = float(y2 - y1)
                det.bbox = bbox
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(cls_id))
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                out.detections.append(det)
            self.publisher.publish(out)

    rclpy.init(args=args)
    node = AghriYoloDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

