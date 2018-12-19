import numpy
from keras.datasets import mnist
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import cv2
import math
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
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
print (X_train.shape[0],X_train.shape[1],X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train/=255
X_test/=255
# one hot encode
number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)
# create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))
# load weights  #model.load_weights("weights.best.hdf5") 
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#save checkpoint      
filepath="/home/diptanshu/Documents/MNIST/test.hdf5"  
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=100,callbacks=callbacks_list)
# Save the model
#model.save('models/mnistCNN.h5') 
# Final evaluation of the model
metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)
from PIL import Image
for t in range(26):
    #a=Image.open(str(t)+".jpg").convert('L')
    #img = a.resize((28,28))
    # read the image
    gray = cv2.imread(str(t)+".jpg", cv2.IMREAD_GRAYSCALE)
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
    #im=numpy.array(img)
    ime=gray.reshape(1,28,28,1)
    y_pred = model.predict_classes(ime)
    print(y_pred)
    cv2.imshow('sample',gray)
    cv2.waitKey(0)