import tensorflow as tf
import os
import numpy as np
import cv2
import math

modelFile = "/Users/yaremayaremcuk/Documents/dogfind/dogsfindmodel.h5"

model = tf.keras.models.load_model(modelFile)


inputSize = (331,331)

alllabels = np.load("/Users/yaremayaremcuk/Documents/data/alllabels.npy")
categories = np.unique(alllabels)


def prepare_img(img):
    print(img.shape)
    resized = cv2.resize(img, inputSize, interpolation=cv2.INTER_AREA)
    print(resized.shape)
    img_res = np.expand_dims(resized, axis=0)
    print(img_res.shape)
    img_res=img_res/255.0
    print(img_res.shape)

    return img_res

testimg = "/Users/yaremayaremcuk/Documents/dogfind/test/0a8d8dda0e354c0571c8d47600ab39a3.jpg"


img = cv2.imread(testimg)
imgmodel = prepare_img(img)

resultarr = model.predict(imgmodel, verbose=1)
answer = np.argmax(resultarr, axis=1)

print(answer)


text = categories[answer[0]]

print(text)


font = cv2.FONT_HERSHEY_DUPLEX
center = (img.shape[1])/2
cv2.putText(img, text, (math.floor(center)-100,20), font, 1, (209,19,88), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
