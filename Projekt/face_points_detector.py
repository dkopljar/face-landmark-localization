try:
    import cPickle as pickle
except ImportError:  # Python 3
    import pickle
import sys
import cv2

import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet
import theano
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
sys.setrecursionlimit(10000)

class FacePointsClassifier:
    def __init__(self, classifier_pickle_path):
        with open(classifier_pickle_path, 'rb') as f:
            self._net = pickle.load(f)
            self._cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def detect_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        result = []
        faces = self._detect_faces(gray)
        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+h]
            resized_image = cv2.resize(face_img, (96, 96)) 
            X_rect = np.vstack(resized_image) / 255.0
            X_rect = X_rect.astype(np.float32)
            X_rect = X_rect.reshape(-1, 1, 96, 96)
            face_points = self._transform_face_points(self._net.predict(X_rect), x, y, face_img.shape[0]/96.0)
            result.append((x, y, w, h, face_points))
        return result

    def _transform_face_points(self, points, start_x, start_y, scale):
        points = points[0]
        xAxis = points[0::2]
        yAxis = points[1::2]
        points = []
        for (x, y) in zip(xAxis, yAxis):
            x = int((x * 48 + 48) * scale + start_x)
            y = int((y * 48 + 48) * scale + start_y)
            points.append((x, y))
        return points

    def _detect_faces(self, img):
        rects = self._cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(100, 100),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        return rects
    