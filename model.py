# Used to train the model.
# Fix error with TF and Keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import csv
import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
tf.python.control_flow_ops = tf
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


width = 320
height = 160
newWidth = height
newHeight = width

# Open the data from the files
def LoadImages(log, index, cols, rows):
    # print(log[index].split('/')[1])
    # need to swap the / to a \ for Windows.
    # print(log[index].split('/')[1])
    img = Image.open("./IMG/%s" % log[index].split('/')[1])
    img = np.array(img)
    #img = img[50:160, 0:320]
    # return img
    # Resizing the picture tends to make the model train better.
    return cv2.resize(img, (newWidth, newHeight))


X = []
y = []
steeringAdjust = 0.3

# open log file
# print("Load .csv file")
logs = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        logs.append(row)

for row in logs:
    X.append(LoadImages(row, 0, width, height))
    y.append(float(row[3]))
    X.append(LoadImages(row, 1, width, height))
    y.append(float(row[3]) + steeringAdjust)
    X.append(LoadImages(row, 2, width, height))
    y.append(float(row[3]) - steeringAdjust)

xTrain = np.array(X)
yTrain = np.array(y)


print("Images and labels loaded")
"""
X, y = shuffle(X, y)
xTrain = X
yTrain = y
# print (xTrain[1].shape)
"""
#image=xTrain[1]#.squeeze()
#plt.figure(figsize=(10,10))
#plt.imshow(image)
# print(yTrain[1])
# image processing.  Flipping images, adding shadows,


def ProcessImage(image, steerAngle):
    # generate a random number,
    num = np.random.randint(3)
    if steerAngle != 0:
        if num == 0:
        # only want to flip the image if it is on a corner
            if steerAngle != 0:
                newImage, newSteering = FlipImage(image, steerAngle)
                return newImage, newSteering
    # This will randomly brighten or darken the image.
    elif num == 1:
        newImage = GammaAdjust(image)
        return newImage, steerAngle
    # default,
    return image, steerAngle


def FlipImage(image, steerAngle):
    flippedImage = cv2.flip(image, 1)
    flippedSteering = -steerAngle
    return flippedImage, flippedSteering

# brighen or darken the image
# http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def GammaAdjust(image):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    gamma = 0.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

index = 0
xProcessed = []
yProcessed = []
for image in X:
    newImage, newSteerAngle = ProcessImage(image, yTrain[index])
    index += 1
    xProcessed.append(newImage)
    yProcessed.append(newSteerAngle)

xProcessed = np.array(xProcessed)
yProcessed = np.array(yProcessed)



#NVIDIA Model
#https://arxiv.org/pdf/1604.07316v1.pdf

model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(newHeight, newWidth,   3), dim_ordering='tf'))
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(newHeight, newWidth,   3)))
model.add(Convolution2D(24,5,5,  dim_ordering='tf', ) ) # input_shape=(newHeight, newWidth, 3),
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(36,5,5, dim_ordering='tf'))
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5, dim_ordering='tf'))
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3, dim_ordering='tf'))
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Activation('softmax'))

model.summary()

model.compile('adam', 'mean_squared_error', ['mean_absolute_error'])
# print("begin training")
history = model.fit(xProcessed, yProcessed, batch_size=128, nb_epoch=20, validation_split=0.2)

#model.save("model.h5")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

print("Model Saved")
