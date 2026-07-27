import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from message_filters import TimeSynchronizer, Subscriber
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

from custom_msgs.msg import MarkerArrayStamped, Float32MultiArrayStamped

from late_fusion_scripts.proyector import Proyector
from late_fusion_scripts.cost_function import iou_2d
from late_fusion_scripts.matching import linear_assignment

import numpy as np


class LateFusionNode(Node):

    def __init__(self):
        super().__init__("late_fusion_node")

        self.proyector = None

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)

        # DECLARE SUBSCRIPTIONS

        self.declare_parameter('lidar_detections_topic', '/detected_bonding_boxes')
        lidar_detections_topic = self.get_parameter('lidar_detections_topic').value

        self.declare_parameter('image_detections_topic', '/yolo_bounding_boxes')
        image_detections_topic = self.get_parameter('image_detections_topic').value

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Detection2DArray, image_detections_topic),
                    Subscriber(self, MarkerArrayStamped, lidar_detections_topic)
                    ],
                queue_size=10,
                )

        ts.registerCallback(self._main_pipeline)

        self.declare_parameter('calibration_topic', '/calibration')
        calibration_topic = self.get_parameter('calibration_topic').value
        self.create_subscription(String, calibration_topic, self._calib_callback, 1)

        # DECLARE PUBLISHERS

        self.declare_parameter('fussed_publisher_topic', '/output')
        fussed_publisher_topic = self.get_parameter('fussed_publisher_topic').value

        self.declare_parameter('unmatched_3d_publisher_topic', '/output')
        unmatched_3d_publisher_topic = self.get_parameter('unmatched_3d_publisher_topic').value

        self.declare_parameter('unmatched_2d_publisher_topic', '/output')
        unmatched_2d_publisher_topic = self.get_parameter('unmatched_2d_publisher_topic').value


        self.fussed_publisher = self.create_publisher(Float32MultiArrayStamped, fussed_publisher_topic, 10)
        self.unmatched_3d_publisher = self.create_publisher(Float32MultiArrayStamped, unmatched_3d_publisher_topic, 10)
        self.unmatched_2d_publisher = self.create_publisher(Float32MultiArrayStamped, unmatched_2d_publisher_topic, 10)

        # OBTAIN ANOTHER DATA
        self.declare_parameter('image_width', 1242)
        self.image_width = self.get_parameter('image_width').value

        self.declare_parameter('image_height', 375)
        self.image_height = self.get_parameter('image_height').value

        self.declare_parameter('iou_threshold', 0.1)
        self.iou_threshold = self.get_parameter('iou_threshold').value

        self.get_logger().info("DeepFussion node up and running...")

    def _calib_callback(self, msg):

        if not self.proyector:
            for line in str(msg).split(r'\n'):
                if line.startswith('std_msgs'):
                    values = line.split("'")[1].split(':')[1].split()
                    P = np.array([float(val) for val in values])
                elif line.startswith('R_rect'):
                    values = [val for val in line.split()]
                    R0 = np.array([float(val) for val in values[1:]])
                elif line.startswith('Tr_velo_cam'):
                    values = [val for val in line.split()]
                    V2C = np.array([float(val) for val in values[1:]])

            P, R0, V2C = P.reshape(3, 4), R0.reshape(3, 3), V2C.reshape(3, 4)

            self.proyector = Proyector(P, R0, V2C, self.image_width, self.image_height)
            self.get_logger().info("Transformation matrix built")

    def _detections2d_to_2dbboxes(self, detections_2d):
        '''args: detections_2d -> detection2darray
        return: image_2dbboxes:(M,4) - 2D bounding boxes from camera in [x1, y1, x2, y2]'''
        bboxes = []

        for det in detections_2d.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            bboxes.append([x1, y1, x2, y2])

        return np.array(bboxes, dtype=np.float32) 

    def _detections3d_to_2dbboxes(self, detections_3d):
        '''args: detections3d -> markerarraystamped(MarkerArray+Header)
        return: lidar_2dbboxes:      (N,4) - 2D bounding boxes projected from 3D detection'''
        detections_2d = self.proyector.proyect(detections_3d)
        return detections_2d

    def _detections3d_to_3dbboxes(self, marker_array):
        """
        Convert MarkerArray with CUBE markers to 3D bounding box params.

        Args:
            marker_array (visualization_msgs.msg.MarkerArray): Input marker array

        Returns:
            numpy.ndarray: Array of shape [n, 7] where each row is:
                           [x, y, z, rot_y, l, w, h]
        """

        markers = marker_array.markers
        bboxes = []

        for i, marker in enumerate(markers.markers):

            # Extract pose information
            pos_x = marker.pose.position.x
            pos_y = marker.pose.position.y
            pos_z = marker.pose.position.z

            # Extract quaternion orientation
            qx = marker.pose.orientation.x
            qy = marker.pose.orientation.y
            qz = marker.pose.orientation.z
            qw = marker.pose.orientation.w

            # Extract scale (dimensions)
            length = marker.scale.x  # x-direction (forward)
            width  = marker.scale.y  # y-direction (left)
            height = marker.scale.z  # z-direction (up)

            # Quaternion → yaw (rot_y)
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            rot_y = math.atan2(siny_cosp, cosy_cosp)

            # Append bbox in format [x, y, z, rot_y, l, w, h]
            bboxes.append([pos_x, pos_y, pos_z, rot_y, length, width, height])

        return np.array(bboxes, dtype=np.float32)

    def _get_meta_from_3ddetections(self, detections3d):
        '''args: detections3d -> markerarraystamped(MarkerArray+Header)
        return: (N,7) - e.g. orientation, detection scores, object type, etc.'''

        meta = []
        marker_array = detections3d.markers
        
        for i, marker in enumerate(marker_array.markers):

            qx = marker.pose.orientation.x
            qy = marker.pose.orientation.y
            qz = marker.pose.orientation.z
            qw = marker.pose.orientation.w

            # Score (si no está definido, ponemos 1.0)
            score = marker.color.a if marker.color.a > 0.0 else 1.0

            # Tipo de objeto (namespace o texto -> convertimos a entero/hash)
            if marker.text:
                type_id = hash(marker.text) % (10**6)  # hash reducido
            else:
                type_id = hash(marker.ns) % (10**6) if marker.ns else -1

            # Marker ID
            marker_id = marker.id

            meta.append([qx, qy, qz, qw, score, type_id, marker_id])

        return np.array(meta, dtype=np.float32)

    def _main_pipeline(self, image_detections, lidar_detections):

        if not self.proyector:
            return

        self.get_logger().info("All data received")

        # Message integrity verification
        
        if not image_detections.detections:
            return

        if not lidar_detections.markers:
            return

        # Preprocessing
        image_2dbboxes = self._detections2d_to_2dbboxes(image_detections) 
        lidar_2dbboxes = self._detections3d_to_2dbboxes(lidar_detections)
        lidar_3dbboxes = self._detections3d_to_3dbboxes(lidar_detections)
        meta_info_array = self._get_meta_from_3ddetections(lidar_detections)

        if lidar_2dbboxes.shape[0] == 0:
            return

        self.get_logger().info("All data correct")

        # Processing
        fused, unmatched_3d, unmatched_2d = self._fuse(
                image_2dbboxes, 
                lidar_2dbboxes, 
                lidar_3dbboxes, 
                meta_info_array
                )

        # Publishing
        now = self.get_clock().now().to_msg()
        self.publish_array(self.fussed_publisher, fused, now, key="dets_3d_fusion")
        self.publish_array(self.unmatched_3d_publisher, unmatched_3d, now, key="dets_3d_only")
        self.publish_array(self.unmatched_2d_publisher, unmatched_2d, now)

        self.get_logger().info(
            f"Fusion:\n Fused \t= {len(fused['dets_3d_fusion'])}, "
            f"Unmatched_3d \t= {len(unmatched_3d['dets_3d_only'])}"
            f"Unmatched_2d \t= {len(unmatched_2d)}"
            )

    def publish_array(self, publisher, data, now, key=None):

        msg = Float32MultiArrayStamped()
        msg.header.stamp = now

        values = data.get(key, []) if key else data

        msg.data = [float(x) for row in values for x in row]

        publisher.publish(msg)

    def _fuse(self, image_2dbboxes, lidar_2dbboxes, lidar_3dbboxes, meta_info_array):
        """
        :param lidar_3dbboxes:  (N,7) - 3D bounding box in camera coords: [x,y,z,rot_y,l,w,h] 
        :param image_2dbboxes:          (M,4) - 2D bounding boxes from camera in [x1, y1, x2, y2]
        :param lidar_2dbboxes:      (N,4) - 2D bounding boxes projected from 3D detection
        :param meta_info_array:       (N,7) - e.g. orientation, detection scores, object type, etc.
        :return:
            detection_3D_fusion: { 'dets_3d_fusion': [...], 'dets_3d_fusion_info': [...] }
            detection_3D_only:   { 'dets_3d_only': [...], 'dets_3d_only_info': [...] }
            image_2dbboxes_only:   [ [...], [...], ... ]
        """
        if len(image_2dbboxes) == 0 or len(lidar_2dbboxes) == 0:
            # If no 2D or no 3Dto2D, then either everything is unmatched or...
            detection_3D_fusion = {'dets_3d_fusion': [], 'dets_3d_fusion_info': []}
            detection_3D_only = {
                'dets_3d_only': lidar_3dbboxes.tolist(),
                'dets_3d_only_info': meta_info_array.tolist()
            } if len(lidar_2dbboxes) > 0 else {'dets_3d_only': [], 'dets_3d_only_info': []}
            image_2dbboxes_only = image_2dbboxes.tolist() if len(image_2dbboxes) > 0 else []
            return detection_3D_fusion, detection_3D_only, image_2dbboxes_only

        # Construct IoU matrix
        iou_matrix = np.zeros((len(image_2dbboxes), len(lidar_2dbboxes)), dtype=np.float32)
        for i, det2d in enumerate(image_2dbboxes):
            for j, det3d2d in enumerate(lidar_2dbboxes):
                iou_matrix[i, j] = iou_2d(det2d, det3d2d)

        # Hungarian / linear assignment
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty((0, 2))

        matched = []
        unmatched_2d = []
        unmatched_3dto2d = []

        for d in range(len(image_2dbboxes)):
            if d not in matched_indices[:, 0]:
                unmatched_2d.append(d)
        for t in range(len(lidar_2dbboxes)):
            if t not in matched_indices[:, 1]:
                unmatched_3dto2d.append(t)
        # filter out any low iou matches
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_2d.append(m[0])
                unmatched_3dto2d.append(m[1])
            else:
                matched.append(m.reshape(1,2))
        if len(matched) == 0:
            matched = np.empty((0,2), dtype=int)
        else:
            matched = np.concatenate(matched, axis=0)

        # Prepare final outputs
        image_2dbboxes_fusion = []
        lidar_2dbboxes_fusion = []
        detection_3D_fusion_vals = []
        detection_3D_fusion_info = []

        for (d_2d_idx, d_3d_idx) in matched:
            image_2dbboxes_fusion.append(image_2dbboxes[d_2d_idx].tolist())
            lidar_2dbboxes_fusion.append(lidar_2dbboxes[d_3d_idx].tolist())
            detection_3D_fusion_vals.append(lidar_3dbboxes[d_3d_idx].tolist())
            detection_3D_fusion_info.append(meta_info_array[d_3d_idx].tolist())

        detection_3D_fusion = {
            'dets_3d_fusion': detection_3D_fusion_vals,
            'dets_3d_fusion_info': detection_3D_fusion_info
        }

        image_2dbboxes_only = [image_2dbboxes[i].tolist() for i in unmatched_2d]
        detection_3D_only_vals = []
        detection_3D_only_info = []
        for idx in unmatched_3dto2d:
            detection_3D_only_vals.append(lidar_3dbboxes[idx].tolist())
            detection_3D_only_info.append(meta_info_array[idx].tolist())
        detection_3D_only = {
            'dets_3d_only': detection_3D_only_vals,
            'dets_3d_only_info': detection_3D_only_info
        }

        return detection_3D_fusion, detection_3D_only, image_2dbboxes_only


def main(args=None) -> None:
    """
    ROS 2 main entrypoint.
    """
    rclpy.init(args=args)
    node = LateFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
