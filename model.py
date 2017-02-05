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

cols = 32
rows = 16


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

print('features normalized')


def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


X_normalized = normalize_grayscale(xTrain)
print("Done")


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(16, 32, 3)))
# print('conv 1', model.output_shape)
# Max pooling will cut your shape in 1/2
model.add(MaxPooling2D((2, 2), dim_ordering='tf'))
# print('conv1 pool',model.output_shape)
model.add(Dropout(0.5))
# print('conv1 dropout',model.output_shape)
model.add(Activation('relu'))
# print('conv1 activation', model.output_shape)
model.add(Flatten())
# print('Flatten', model.output_shape)
model.add(Dense(128))
# print('FC1',model.output_shape)
model.add(Activation('relu'))
# print('FC1 Activation',model.output_shape)
model.add(Dense(1))
# print('FC2',model.output_shape)
#model.add(Activation('softmax'))
#print('FC 2 activation',model.output_shape)

model.summary() # print model summary
model.compile('adam', 'mean_squared_error', ['accuracy'])
print("begin training")
history = model.fit(xTrain, yTrain, nb_epoch=40, validation_split=0.2)