import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import time
import random
import onnxruntime as ort


def scaleImage(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        scale_ratio = min(scale_ratio, 1.0)

    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio)) 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # divide padding into 2 sides
    dw /= 2
    dh /= 2   

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add border 
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, scale_ratio, (dw, dh) 


class ObjectDetector(Node):
    def __init__(self, onnx_path, debug=False):
        super().__init__('object_detector')
        self.cv_bridge = CvBridge()

        # Initialize Subcriber
        self.rgb_sub = Subscriber(self ,Image, 'camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

        filter_msg = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                                      queue_size=10, slop =1)
        
        filter_msg.registerCallback(self.detection_callback)

        # Publish Bounding Boxes
        self.pub_bboxes = self.create_publisher(BoundingBoxes, "yolov7_detection/bounding_boxes", 10)
        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=self.provider)
        self.names = ['Cabinet', 'Chair', 'Person', 'Table']
        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.names)} 
        self.debug = debug 

    def detection_callback(self, image,depth):
        self.get_logger().warning("Working RGBD xD")
        out_msg = BoundingBoxes()

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image, 'bgr8')
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error('Error processing RGBD image: {0}'.format(e))

        cv_image = np.asanyarray(cv_image)
        cv_depth = np.asanyarray(cv_depth)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        image = img.copy()
        image, ratio, dwdh = scaleImage(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255 
        
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        out = self.session.run(outname, inp)[0]
        out_msg = BoundingBoxes()  
        
        for (batch_id, x1, y1, x2, y2, cls_id, score) in out:
            bbox = BoundingBox()
            box = np.array([x1, y1, x2, y2]) 
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist() 
            # ===== Distance object =========
            x1, y1, x2, y2 = box
            depth = cv_depth.copy()
            depth = depth[y1:y2, x1:x2].astype(float)
            depth_crop = depth.copy()
            if depth_crop.size == 0:
                continue
            depth_res = depth_crop[depth_crop != 0]

            # Get data scale from the device and convert to meters
            depth_res = depth_res * 0.0010000000474974513
            if depth_res.size == 0:
                continue
            # dist is distance of object 
            dist = np.mean(depth_res)
            dist_3demical = round(dist,3)  #get distance with two demical points

            #======== prepare data for publish ========
            bbox.xmin = max(int(box[0]), 0)
            bbox.ymin = max(int(box[1]), 0)
            bbox.xmax = max(int(box[2]), 0)
            bbox.ymax = max(int(box[3]), 0)
            bbox.class_id = self.names[int(cls_id)]
            bbox.probability = float(score)
            bbox.center_dist = float(dist_3demical)
            bbox.img_height = int(img_height)
            bbox.img_width = int(img_width)
            bbox.class_id_int = int(cls_id)
            out_msg.bounding_boxes.append(bbox)

        #****** Start Publish Data *******
        self.pub_bboxes.publish(out_msg)
        print(out_msg.bounding_boxes)


def main(args=None):
    print('object_detection.')
    rclpy.init(args=args) 
    detector = ObjectDetector(onnx_path="/home/muchiro/indoor_detection.onnx")
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
