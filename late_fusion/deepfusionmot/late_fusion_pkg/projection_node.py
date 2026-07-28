import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import TimeSynchronizer, Subscriber

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from late_fusion_scripts.proyector import Proyector
from custom_msgs.msg import  Float32MultiArrayStamped

class ProjectionNode(Node):

    def __init__(self):
        super().__init__("projection_node")

        self.proyector = None

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)

        # DECLARE SUBSCRIPTIONS

        self.declare_parameter('image_input_topic', '/image_raw')
        image_input_topic = self.get_parameter('image_input_topic').value

        self.declare_parameter('fused_detections_topic', '/fused/detections')
        fused_detections_topic = self.get_parameter('fused_detections_topic').value

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_input_topic),
                    Subscriber(self, Float32MultiArrayStamped, fused_detections_topic)
                    ],
                queue_size=10,
                )

        ts.registerCallback(self._main_pipeline)

        self.declare_parameter('calibration_topic', '/calibration')
        calibration_topic = self.get_parameter('calibration_topic').value
        self.create_subscription(String, calibration_topic, self._calib_callback, 1)

        self.declare_parameter('output_image_topic', '/yolo/image')
        output_image_topic = self.get_parameter('output_image_topic').value
        self.publisher = self.create_publisher(Image, output_image_topic, 1)

        self.bridge = CvBridge()

        self.get_logger().info("Proyection node up and running")

    def detections_3d_to_2d(self, cv_image, fused_detections):
        self.proyector.draw_proyections(cv_image, fused_detections)

    def _main_pipeline(self, image_msg, fused_detections):

        cv_image = self._imgmsg2np(image_msg)
        image_with_proyections = self.detections_3d_to_2d(cv_image, fused_detections)

        image_msg = self._np2imgmsg(image_with_proyections)

        self.publisher.publish(image_msg)

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to a numpy array.

        Args:
            img_msg (Image): Image message.

        Returns:
            np.ndarray: OpenCV image.
        """
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        """
        Convert a numpy array to a sensor_msgs/Image message.

        Args:
            arr (np.ndarray): OpenCV image.

        Returns:
            Image: Image message.
        """
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

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


def main(args=None) -> None:
    """
    ROS 2 main entrypoint.
    """
    rclpy.init(args=args)
    node = ProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
