
from PIL import Image
import sys
import os
import cv2
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from utils.plot_image import display
from utils.config import Config
from configs.config import CFG_CAMPUS

class CampusInferrer:
    def __init__(self):
        self.config = Config.from_json(CFG_CAMPUS)
        self.classes = None
        self.net = None
        self._layer_names=None
        self._output_layers=None
        self.image_blob=None
        self.layer_results=None
        self.image = np.asarray(Image.open('/opt/project/weights/yolo_campus/IMG_4381.JPG')).astype(np.float32)

    """self.config.path.obj_name"""
    def Load_classes(self):
        obj_path =  self.config.data.path.obj #"/opt/project/weights/yolo_campus/obj.names"
        with open(obj_path, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def inference_engine(self):
        cfg_path = self.config.data.path.cfg
        weight_path= self.config.data.path.weight
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._layer_names = self.net.getLayerNames()
        self._output_layers = [self._layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def proces(self):
        self.image_blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (480, 480), swapRB=True, crop=False)
        self.net.setInput(self.image_blob, "data")
        self.layer_results = self.net.forward(self._output_layers)


    def final_prediction(self,img):#outputs, img, threshold, nms_threshold #layers_result, img, 0.3, 0.3)
        height, width =  img.shape[0], img.shape[1]  # self.image.shape[0], self.image.shape[1] #
        boxes, confs, class_ids = [], [], []
        final_result = []
        for output in self.layer_results:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)


        merge_boxes_ids = cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.3)
        for i in merge_boxes_ids:
            final_result.append([boxes[int(i)][0], boxes[int(i)][1], boxes[int(i)][2], boxes[int(i)][3],
                                   confs[int(i)], np.int64(class_ids[int(i)]).item()])

        final_result = json.dumps(final_result)
        return final_result
# tek obje json dosyası , config dosyası , postman google dan bak,utilse yeni kendin icin config yaz


c_net = CampusInferrer()
c_net.Load_classes()
c_net.inference_engine()
c_net.proces()
image = np.asarray(Image.open('/opt/project/weights/yolo_campus/IMG_4381.JPG')).astype(np.float32)
print(c_net.final_prediction(image))










"""




import cv2
import numpy as np


# Specify your models network size
network_size = (480, 480)
# Darknet cfg file path
cfg_path = '/opt/project/weights/yolo_campus/yolov3.cfg'
# Darknet weights path
weights_path = '/opt/project/weights/yolo_campus/yolov3_final.weights'

# Load names of classes
classesFile = "/opt/project/weights/yolo_campus/obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Define the inference engine
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
_layer_names = net.getLayerNames()
_output_layers = [_layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Read Image as input
img_path = '/opt/project/weights/yolo_campus/img_1.jpg'
img = cv2.imread(img_path)
image_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, network_size, swapRB=True, crop=False)
net.setInput(image_blob, "data")

# Run inference
layers_result = net.forward(_output_layers)


# Convert layers_result to bbox, confs and classes
def get_final_predictions(outputs, img, threshold, nms_threshold):
    height, width = img.shape[0], img.shape[1]
    boxes, confs, class_ids = [], [], []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    merge_boxes_ids = cv2.dnn.NMSBoxes(boxes, confs, threshold, nms_threshold)

    # Filter only the boxes left after nms
    boxes = [boxes[int(i)] for i in merge_boxes_ids]
    confs = [confs[int(i)] for i in merge_boxes_ids]
    class_ids = [class_ids[int(i)] for i in merge_boxes_ids]
    return boxes, confs, class_ids


boxes, confs, class_ids = get_final_predictions(layers_result, img, 0.3, 0.3)
print(classes[class_ids[0]])
print(confs)
print(boxes)

"""