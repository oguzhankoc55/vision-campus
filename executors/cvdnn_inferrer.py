import cv2
import numpy as np

# Specify your models network size
network_size = (480, 480)
# Darknet cfg file path
cfg_path = '/darknet/cfg/yolov3-tiny.cfg'
# Darknet weights path
weights_path = '/opt/project/weights/yolo/yolov3-tiny.weights'

# Load names of classes
classesFile = "/darknet/cfg/coco.names"
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
img_path = '/darknet/data/dog.jpg'
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