# test server
import socket
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import threading
import tensorflow as tf
import daemon
import os

# create a socket for serving
def create_server(host,port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((host,port))
    return server_socket

def get_connection_and_params(server_socket,sleep_time):
    print 'listening for connection'
    server_socket.listen(10)
    while True:
        try:
            conn,addr = server_socket.accept()
            print 'waiting to accept'
            break
        except:
            time.sleep(sleep_time)
            continue
    print 'connected at: ' + str(addr)
    data_params = ''
    while True:
        try:
            new_data=conn.recv(1024)
            data_params = data_params + new_data
            if len(data_params) > 0:
                if data_params[len(data_params)-1] == '\r': break
        except:
            time.sleep(sleep_time)
            continue
    data_params = data_params[0:len(data_params)-1]
    data_params = data_params.split('\n')
    return conn,addr,data_params

def load_test_data():
    global x_train
    global x_test
    batch_size = 128
    num_classes = 10
    epochs = 12
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # channels first or channels last determines dimension labels
    # tensorflow is usually channels last, while theano is channels first
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    # get format of data correct
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

# setup tensorflow for threading of different models
def tensorflow_setup():
    global sess
    global model
    global model2
    global default_graph
    global lock
    
    # setup tensorflow session
    sess = tf.Session()
    K.set_session(sess)
    model = load_model('/home/npropes/Desktop/mnist-tf1.h5') 
    model2 = load_model('/home/npropes/Desktop/mnist-tf2.h5')
    default_graph = tf.get_default_graph()

    # set up a global lock to control access to tf session
    lock = threading.Lock()

# compute prediction for a model
# uses lock to control access to shared resource (tf session)
def set_session(i,m):
    lock.acquire()  
    with default_graph.as_default():
        pred = m.predict(x_test[i:i+1,:,:,:])[0][0]
    lock.release()
    return pred 

# test thread for model #1
def test_thread(n,c):
    for i in range(0,5000):
        c.sendall(n + ' ' + str(set_session(i,model)))
    c.sendall('finished\r')
    c.close()

# test thread for model #2
def test_thread2(n,c):
    for i in range(0,5000):
        c.sendall(n + ' ' + str(set_session(i,model2)))
    c.sendall('finished\r')
    c.close()

# remove dead threads
def cleanup_threads(t):
    for i in range(len(t)-1,-1,-1):
        if not t[i].is_alive():
            t.pop(i)

def thread_maker():
    # load test data
    load_test_data()
    
    # setup tensorflow and models
    tensorflow_setup()

    # create a server
    host = ''
    port = 8999
    s = create_server(host,port)

    # list of threads
    threads = []

    # main while loop
    while True:
        print 'hi'
        # get connection and params 
        conn,addr,params = get_connection_and_params(s,1)
	# parse param list
        modeltype = str(params[0])
        # remove dead threads
        cleanup_threads(threads)
        # create thread based on model type
        if modeltype == 'model1':
            threads.append(threading.Thread(target = test_thread, args=('m1-t' + str(len(threads)),conn,)))
            threads[len(threads)-1].setDaemon(True)
        elif modeltype == 'model2':
            threads.append(threading.Thread(target = test_thread2, args=('m2-t' + str(len(threads)),conn,)))
            threads[len(threads)-1].setDaemon(True)
        threads[len(threads)-1].start()

if __name__ == '__main__':
    with daemon.DaemonContext():
        thread_maker()



