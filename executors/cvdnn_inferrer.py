

import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from utils.plot_image import display

from utils.config import Config
from configs.config import CFG_CAMPUS


class Campus_inferrer:
    def __init__(self):
        self.config = CFG_CAMPUS
        self.classes = None
        self.net = None
        self._layer_names=None
        self._output_layers=None
        self.image_blob=None

    def Load_classes(self):
        with open(self.config.path.obj_name, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def def_interface_engine(self):
        self.net = cv2.dnn.readNetFromDarknet(self.config.path.cfg, self.config.path.weight)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._layer_names = self.net.getLayerNames()
        self._output_layers = [self._layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def proces(self, image=None):
        self.image_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.config.netwotk_size, swapRB=True, crop=False)
        self.net.setInput(self.image_blob, "data")
        return self.net.forward(self._output_layers)

    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.proces(image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image, [1, shape[0], shape[1], shape[2]])
        print(tensor_image.shape)
        #pred = self.predict(tensor_image)['conv2d_transpose_4']
        #display([tensor_image[0], pred[0]])
        pred = tensor_image.numpy().tolist()
        return {'segmentation_output':pred}














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