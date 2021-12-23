import requests
from PIL import Image
import numpy as np
import os
import sys
from executors.campus_inferrer import CampusInferrer
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

ENDPOINT_URL = "http://127.0.0.1:5000/infer"

c_net = CampusInferrer()

def infer():
    #image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32)
    image = np.asarray(Image.open('weights/yolo_campus/IMG_4381.JPG')).astype(np.float32)
    data ={'image': image.tolist()}
    c_net.Load_classes()
    c_net.inference_engine()
    c_net.proces()
    response = c_net.final_prediction(image)
    #response = requests.post(ENDPOINT_URL, json = data)
    print(response.raise_for_status())
    print(response.json())


if __name__ =="main":
    infer()


"""import requests
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(file),'../'))

ENDPOINT_URL = "http://127.0.0.1:5000/infer"

def infer():
    #image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32)
    image = np.asarray(Image.open('weights/yolo_campus/IMG_4381.JPG')).astype(np.float32)
    data ={'image': image.tolist()}
    response = requests.post(ENDPOINT_URL, json = data)
    print(response.raise_for_status())
    print(response.json())


if name =="main":
    infer()
"""