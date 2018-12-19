# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import math
from scipy import ndimage

K.set_image_dim_ordering('th')
import cv2
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = numpy.round(cols/2.0-cx).astype(int)
    shifty = numpy.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
def shift(img,sx,sy):
    rows,cols = img.shape
    M = numpy.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)
# Final evaluation of the model
model.save('models/mnistCNN.h5')
'''for no in range (26):
    # read the image
    gray = cv2.imread(str(no)+".jpg", cv2.IMREAD_GRAYSCALE)
    # resize the images and invert it (black background)
    gray = cv2.resize(255-gray, (28, 28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # save the processed images
    while numpy.sum(gray[0]) == 0:
        gray = gray[1:]
    while numpy.sum(gray[:,0]) == 0:
        gray = numpy.delete(gray,0,1)
    while numpy.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while numpy.sum(gray[:,-1]) == 0:
        gray = numpy.delete(gray,-1,1)
    rows,cols = gray.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows)) 
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = numpy.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted       
    cv2.imwrite("pro-img/image_"+str(no)+".jpg", gray)
    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    #images[i] = flatten
    #correct_val = numpy.zeros((10))
    ''''''cv2.imshow('in',gray)
    cv2.waitKey(0)
    d=int(input('enter number shown in image'))
    correct_val[d] = 1
    correct_vals[i] = correct_val''''''
    #i += 1
"""
the prediction will be an array with four values,
which show the predicted number
"""

#prediction = tf.argmax(y,1)
"""
we want to run the prediction and the accuracy function
using our generated arrays (images and correct_vals)
"""'''
from PIL import Image
for t in range(26):
    a=Image.open(str(t)+".jpg").convert('L')
    img = a.resize((28,28))
    img.show()
    im=numpy.array(img)
    im=im.reshape(1,1,28,28,)
    y_pred = model.predict(im)
    print(y_pred)