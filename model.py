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
from keras.layers import Convolution2D, Lambda
from keras.layers.pooling import MaxPooling2D
tf.python.control_flow_ops = tf
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')


cols = 160
rows = 80


# Open the data from the files
def LoadImages(log, index, cols, rows):
    # print(log[index].split('/')[1])
    # need to swap the / to a \ for Windows.
    # print(log[index].split('/')[1])
    img = Image.open("./IMG/%s" % log[index].split('/')[1])
    img = np.array(img)
    return cv2.resize(img, (cols, rows))


X = []
y = []
steeringAdjust = 0.3

# open log file
print("Load .csv file")
logs = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        logs.append(row)

for row in logs:
    X.append(LoadImages(row, 0, cols, rows))
    y.append(float(row[3]))
    X.append(LoadImages(row, 1, cols, rows))
    y.append(float(row[3]) + steeringAdjust)
    X.append(LoadImages(row, 2, cols, rows))
    y.append(float(row[3]) + steeringAdjust)

X = np.array(X)
y = np.array(y)


print("Images and labels loaded")


# preprocessing
# Load the data into training and test sets
xTrain = X[:-2000]
yTrain = y[:-2000]
xTest = X[-2000]
yTest = y[-2000]
print("Train totals", xTrain.shape)
print("Test totals", xTest.shape)



print (xTrain[1].shape)

image=xTrain[1]#.squeeze()
plt.figure(figsize=(10,10))
plt.imshow(image)
print(yTrain[1])


# One-Hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(yTrain)


def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


X_normalized = normalize_grayscale(xTrain)
print("Done")

"""
# Original Model, was not learning, so i tried to implement the NVidia model below.
model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=(rows, cols, 3), dim_ordering='tf'))
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
model.add(Dropout(0.5))
model.add(Activation('relu'))

#model.add(Convolution2D())

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('softmax'))
model.summary() # print model summary
"""
#NVIDIA Model
#https://arxiv.org/pdf/1604.07316v1.pdf

model = Sequential()

model.add(Convolution2D(24,5,5, input_shape=(rows,cols,3), dim_ordering='tf'))
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
#model.add(Activation('softmax'))



model.summary()



model.compile('adam', 'mean_squared_error', ['accuracy'])
print("begin training")
history = model.fit(xTrain, yTrain, nb_epoch=5, validation_split=0.2)

