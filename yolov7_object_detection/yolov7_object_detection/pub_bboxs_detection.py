import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import time
import onnxruntime


class_names = ['Cabinet', 'Chair', 'Person', 'Table']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = x[0] + x[2] / 2
    y[1] = x[1] + x[3] / 2
    y[2] = x[2] - x[0] 
    y[3] = x[3] - x[1] 
    x_mid = y[0]
    y_mid = y[1]
    w = y[2]
    h = y[3]
    return x_mid , y_mid, w , h




class YOLOv7:

    def __init__(self, path, conf_thres=0.25, iou_thres=0.45, official_nms=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.official_nms = official_nms

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names or self.official_nms


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        #Pinto's postprocessing is different from the official nms version
        if self.official_nms:
            scores = outputs[0][:,-1]
            predictions = outputs[0][:, [0,5,1,2,3,4]]
        else:
            scores = np.squeeze(outputs[0], axis=1)
            predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        if not self.official_nms:
            boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class MySubscriber(Node):
    def __init__(self, model_path):
        super().__init__('subscriber_RGB_Depth')
        # Bridge from ROS to OpenCV
        self.cv_bridge = CvBridge()

        # Initialize Model YOLOv7
        self.yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

        # Initialize Subcriber
        self.rgb_sub = Subscriber(self ,Image, 'camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

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

        cv_image = np.asanyarray(cv_image)
        cv_depth = np.asanyarray(cv_depth)

        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

        # process detect object 
        boxes, scores, class_ids = self.yolov7_detector(cv_image)

        mask_alpha=0.3
        mask_img = cv_image.copy()
        det_img = cv_image.copy()

        img_height, img_width = cv_image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # bounding boxes and labels of detections
        for box,score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)
            x = (x2 + x1) /2
            y = (y2 + y1)/2
            w = x2 - x1 
            h = y2 - y1
            x, y, w, h = int(x), int(y), int(w), int(h)
            depth = cv_depth.copy()
            depth = depth[x:x+w, y:y+h].astype(float)

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
            label = class_names[class_id]
    
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
            bbox.class_id_int = int(class_id)
            out_msg.bounding_boxes.append(bbox)
            
            # ============== Start Ploting ==================

            #draw circle depth
            cv2.circle(det_img, (x,y), 2, color, 2)
            # Draw rectangle
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
            cv2.putText(det_img, text, (x-1,y-1),
                        cv2.FONT_HERSHEY_SIMPLEX,size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        real_time = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
        # ******* start Publish ********
        self.pub_bboxes.publish(out_msg)
        print(out_msg.bounding_boxes)
        cv2.imshow("Detected Objects", real_time)
        # Press key q to stop
        cv2.waitKey(1) & 0xFF == ord('q')
        
def main(args=None):
    print('YOLOv7 Object Detection')
    rclpy.init(args=args)
    subscriber = MySubscriber("/home/muchiro/modified_indoor_detection.onnx")
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



