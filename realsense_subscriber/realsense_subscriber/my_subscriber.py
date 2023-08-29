import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import time
import onnxruntime


class MySubscriber(Node):
    def __init__(self):
        super().__init__('my_subscriber')
        self.cv_bridge = CvBridge()
        # self.yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
        self.rgb_subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.rgb_callback,
            10
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

    def rgb_callback(self, msg):
        # Process RGB data here
        self.get_logger().warning("Receving RGB frame")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            #cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
            # Update object localizer
            #boxes, scores, class_ids = self.yolov7_detector(cv_image)
            #combined_img = self.yolov7_detector.draw_detections(cv_image)
            #cv2.imshow("Detected Objects", combined_img)
            # Press key q to stop
            cv2.imshow("RGB",cv_image)
            cv2.waitKey(1) & 0xFF == ord('q')
        except Exception as e:
            self.get_logger().error('Error processing RGB image: {0}'.format(e))

    def depth_callback(self, msg):
        # Process depth data here
        self.get_logger().warning("Receving Depth frame")
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Normalize depth values for visualization
            cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Display the image
            cv2.imshow('Depth Image', cv_image_normalized)
            cv2.waitKey(1)  # Refresh display

        except Exception as e:
            self.get_logger().error('Error processing depth image: {0}'.format(e))

def main(args=None):
    rclpy.init(args=args)
    subscriber = MySubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


