import numpy as np
import cv2
import pandas as pd
import os


img_size = (331, 331)
img_full_size = (331,331,3)

trainimagefolder = "/Users/yaremayaremcuk/Documents/dogfind/train"

df = pd.read_csv("/Users/yaremayaremcuk/Documents/dogfind/labels.csv")

print("Head of lables:")
print("===============")
print(df.head())
print(df.describe())

print("Group by labels:")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))

imgpath = "/Users/yaremayaremcuk/Documents/dogfind/train/0a0c223352985ec154fd604d7ddceabd.jpg"
img = cv2.imread(imgpath)



allimg = []
alllabels = []


for idx , (image_name, breed) in enumerate(df[['id' , 'breed']].values):
    img_path = os.path.join(trainimagefolder, image_name + '.jpg')
    print(img_path)
    if len(allimg) < 8500 and len(alllabels) < 8500:

        img = cv2.imread(img_path)
        resized = cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)
        allimg.append(resized)
        alllabels.append(breed)


print(len(allimg))
print(len(alllabels))

print("save the data: ")

np.save("/Users/yaremayaremcuk/Documents/data/allimages.npy",allimg)
np.save("/Users/yaremayaremcuk/Documents/data/alllabels.npy",alllabels)

print("done")