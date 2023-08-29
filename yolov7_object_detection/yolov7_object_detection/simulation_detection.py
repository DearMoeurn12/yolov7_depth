import enum
from pyexpat import model
import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
import time
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import numpy as np

class MySubscriber(Node):
    def __init__(self, model_path):
        super().__init__('subscriber_RGB_Depth')
        # Bridge from ROS to OpenCV
        self.cv_bridge = CvBridge()

        # Initialize Model YOLOv7
        self.device = select_device('0')
        self.half = self.device.type != 'cpu' 
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(640, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0,255, size=(len(self.names), 3))
        if self.half:
            self.model.half()  # to FP16
        # Initialize Subcriber
        self.rgb_sub = Subscriber(self ,Image, '/depth_camera/image_raw') # For simulation /depth_camera/image_raw
        self.depth_sub = Subscriber(self, Image, '/depth_camera/depth/image_raw')#/depth_camera/depth/image_raw

        filter_msg = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                                      queue_size=10, slop =1)
        
        filter_msg.registerCallback(self.callback)

        # Publish Bounding Boxes
        self.pub_bboxes = self.create_publisher(BoundingBoxes, "yolov7_detection/bounding_boxes", 10)


    def callback(self, image, depth):
        self.get_logger().warning("Working RGBD xD")
        out_msg = BoundingBoxes()
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image, 'bgr8')
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error('Error processing RGBD image: {0}'.format(e))

        cv_depth = np.asanyarray(cv_depth)
        # Pre-process image
        img = letterbox(cv_image, self.imgsz, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Convert image to torch tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        start_time = time.time() 
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.5)

        FPS = "FPS: " + str("{0:}").format(int(1.0 / (time.time() - start_time))) # FPS = 1 / time to process loop
        # process detect object 

        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        mask_alpha=0.2
        mask_img = cv_image.copy()
        det_img = cv_image.copy()

        img_height, img_width = cv_image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

        # bounding boxes and labels of detections
        for i,det in enumerate(pred):
            if len(det):
                det = det.detach()  # detach from the computation graph
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_image.shape).round()
                for *xyxy, score, cls in reversed(det):
                    color = self.colors[int(cls)]
                    xyxy_values = [int(tensor.item()) for tensor in xyxy]
                    x1, y1, x2, y2 = int(xyxy_values[0]), int(xyxy_values[1]), int(xyxy_values[2]), int(xyxy_values[3])
                    depth = cv_depth.copy()
                    depth = depth[y1:y2, x1:x2].astype(float)

                    depth_crop = depth.copy()
                    if depth_crop.size == 0:
                        continue
                    depth_res = depth_crop[depth_crop != 0]

                    # Get data scale from the device and convert to meters
                    depth_res = depth_res * 1
                    if depth_res.size == 0:
                        continue
                    # dist is distance of object 
                    dist = min(depth_res)
                
                    dist_3demical = round(dist,3)  #get distance with two demical points
                    label = self.names[int(cls)]
            
                    # -------------- Extract data for Publisher --------
                    bbox = BoundingBox()
                    box = np.array([x1, y1, x2, y2]) # Bounding Box of object
                    bbox.xmin = max(int(box[0]), 0)
                    bbox.ymin = max(int(box[1]), 0)
                    bbox.xmax = max(int(box[2]), 0)
                    bbox.ymax = max(int(box[3]), 0)
                    bbox.class_id = label
                    bbox.probability = float(score)
                    bbox.center_dist = float(dist_3demical)
                    bbox.img_height = int(img_height)
                    bbox.img_width = int(img_width)
                    bbox.class_id_int = int(cls)
                    out_msg.bounding_boxes.append(bbox)
                    
                    # ============== Start Ploting ==================
                    x_mid =int((x1+x2)/2) 
                    y_mid= int((y1+y2)/2) 
                    #draw circle depth
                    cv2.circle(det_img, (x_mid,y_mid), 2, color, 2)
                    #Draw rectangle
                    cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw fill rectangle in mask image
                    cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

                    
                    caption = f'{label} {int(score * 100)}%'
                    text = "Depth: " + str("{0:.2f}").format(dist) +'Meter'
                    (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=size, thickness=text_thickness)
                    th = int(th * 1.2)
                    
                    cv2.rectangle(det_img, (x1, y1),
                                (x1 + tw, y1 - th), color, -1)
                    cv2.rectangle(mask_img, (x1, y1),
                                (x1 + tw, y1 - th), color, -1)
                    cv2.putText(det_img, caption, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
                    cv2.putText(mask_img, caption, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
                    cv2.putText(det_img, text, (x_mid,y_mid),
                                cv2.FONT_HERSHEY_SIMPLEX,size, (255, 255, 255), text_thickness, cv2.LINE_AA)
                    cv2.putText(det_img, FPS , (4,10),
                                cv2.FONT_HERSHEY_SIMPLEX,size, (1, 1, 255), text_thickness, cv2.LINE_AA)

        real_time = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
        cv2.imshow("Detected Objects", real_time)
        # Press key q to stop
        cv2.waitKey(1) & 0xFF == ord('q')
        # ******* start Publish ********
        self.pub_bboxes.publish(out_msg)
        print(out_msg.bounding_boxes)

        
def main(args=None):
    print('YOLOv7 Object Detection')
    rclpy.init(args=args)
    subscriber = MySubscriber("/home/muchiro/best.pt")
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



