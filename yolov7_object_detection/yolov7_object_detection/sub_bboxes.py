import cv2 
import numpy as np 
import rclpy
from rclpy.node import Node 
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox


class SubscriberObjectDetection(Node):
    def __init__(self):
        super().__init__('Subscriber_Boxes')
        self.sub = self.create_subscription(BoundingBoxes,'yolov7_detection/bounding_boxes', self.callback, 10)

    def callback(self, msg_bbox):
        self.get_logger().warning("Working BoundingBoxes Subscriber xD")
        boxes = msg_bbox.bounding_boxes
        for bbox in boxes:
            print("BoundingBox:")
            print(f"  probability: {bbox.probability}")
            print(f"  xmin: {bbox.xmin}")
            print(f"  ymin: {bbox.ymin}")
            print(f"  xmax: {bbox.xmax}")
            print(f"  ymax: {bbox.ymax}")
            print(f"  img_width: {bbox.img_width}")
            print(f"  img_height: {bbox.img_height}")
            print(f"  center_dist: {bbox.center_dist}")
            print(f"  class_id: {bbox.class_id}")
            print(f"  class_int_id:{bbox.class_id_int}")
            print("]")
            print("========================")  # Add a blank line between bounding boxes


def main(args=None):
    rclpy.init(args=args) 
    Sub_bboxes = SubscriberObjectDetection()
    rclpy.spin(Sub_bboxes)
    Sub_bboxes.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
