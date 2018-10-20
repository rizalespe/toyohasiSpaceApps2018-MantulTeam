# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:08:05 2018

@author: Oskar
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os



# PAKAI FIX DIRECTORY PATH FOR MODEL AND PROTOTXT
#NO COMMAND LINE ARGS
args = {
	"model": "mantul_model1.h5",
    "labelbin" : "lb1.pickle",
    "image": "test_image\horizon_9.png"
}

"""

# pakai command line
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image (test_image)")
args = vars(ap.parse_args())
"""


# load image
#image = cv2.imread(args["image"])
from PIL import Image
import requests
from io import BytesIO
#response = requests.get("https://static.turbosquid.com/Preview/2016/02/25__04_25_50/marssurface_001.jpg545afbfe-350e-41f4-907d-d158e41cda7dOriginal.jpg")
response = requests.get("https://upload.wikimedia.org/wikipedia/en/8/8d/Rocky_Mars_Surface.jpeg")
image = Image.open(BytesIO(response.content))
image = np.array(image)
output = image.copy()
 
# resize
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("load model")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())
 
# classify the input image
print("classify image")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

#true/false?
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=800)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (255, 255, 255), 2)
 
#output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)

