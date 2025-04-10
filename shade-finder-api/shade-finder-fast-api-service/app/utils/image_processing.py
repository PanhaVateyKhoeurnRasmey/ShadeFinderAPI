#@title Import libraries
import base64
import io
import json
import tempfile
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import imutils
from sklearn.cluster import KMeans
import stone
import pickle

from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image
from PIL import Image as ImagePIL
import os

print("image_processing.py, current working directory:", os.getcwd())

#@title Classes and functions
class DominantColors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        # read image
        img_src = self.IMAGE
        # percent by which the image is resized
        scale_percent = 10
        # calculate the 50 percent of original dimensions
        width = int(img_src.shape[1] * scale_percent / 100)
        height = int(img_src.shape[0] * scale_percent / 100)
        # dsize
        dsize = (width, height)
        # resize image
        small_img = cv2.resize(img_src, dsize)
        # convert to rgb from bgr
        img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        # save image after operations
        self.IMAGE = img
        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)
        # the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        # save labels
        self.LABELS = kmeans.labels_
        # returning after converting to integer from float
        return self.COLORS.astype(int)
    def plotHistogram(self):

        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)

        #create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        #appending frequencies to cluster centers
        colors = self.COLORS

        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]

        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end

        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

def removeBlackImg(img):
         # Converting from BGR Colours Space to gray
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         # Global thresholding
        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)

         # Get rgb values from image
        b, g, r = cv2.split(img)

         # Assign new color values to image
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)

         # Store new image in the array
        return dst

def skinDetectImg(img):

            # Read the image
            image = cv2.imread(img)

            # Resize the image
            new_im = imutils.resize(image,width=800, height=800)

            # Call extractSkin function
            skin = extractSkin(new_im)

            return skin

def extractSkin(img):
    # Taking a copy of the image
    #img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def getColorImg(img):
        # Number of colors to be extracted
        clusters = 2

        # Array to store extracted color
        c = []

        # Array to store new colors of all images
        colors = []

        # Call dominantColors to get colors list
        dc = DominantColors(img, clusters)
        colors_list = dc.dominantColors()

        # Store the dominant color if not black
        if (colors_list[0].all() != 0):
            c = colors_list[0]
        else:
            c = colors_list[1]

        return c

def crop_face(img_path):
    image = ImagePIL.open(img_path)
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces = detect_faces(image)
    if len(faces) > 0:
        cropped = image.crop(faces[0].bbox.scale(image.size).as_tuple)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cropped.save(tmp.name)
            return tmp.name, True
    return img_path, False


class UndertonePredictor:
    def __init__(self, model_path='undertone_model.pkl'):
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_tone(self, img_path, tone_palette=None, tone_labels=None):
        # Extract Face
        cropped,_ = crop_face(img_path)
        img_path = cropped

        if tone_palette is not None and tone_labels is not None:
            result = stone.process(img_path, tone_palette=tone_palette, tone_labels=tone_labels)
        elif tone_palette is None and tone_labels is None:
            result = stone.process(img_path)
        else:
            raise Exception("Both tone_palette and tone_labels have to be defined")

        return result['faces'][0]

    def predict_undertone(self, img_path, neutral=0.33):
        # Extract Face
        cropped,_ = crop_face(img_path)
        img_path = cropped

        skin = skinDetectImg(img_path)
        skin_no_black = removeBlackImg(skin)
        colors = getColorImg(skin_no_black)
        undertone_prob = list(self.model.predict_proba(colors.reshape(1,-1))[0])
        if undertone_prob[1] > 1 - neutral:
            return {"undertone":"warm", "score":undertone_prob}
        elif undertone_prob[1] < neutral:
            return {"undertone":"cool", "score":undertone_prob}
        else:
            return {"undertone":"neutral", "score":undertone_prob}