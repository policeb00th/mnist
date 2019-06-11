import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def trn(_):
	import tensorflow as tf
	from numpy import loadtxt
	import matplotlib.pyplot as plt
	import numpy
	from keras import backend as K
	config = tf.ConfigProto(intra_op_parallelism_threads=1,
							inter_op_parallelism_threads=1,
							allow_soft_placement=True)                            
	session = tf.Session(config=config)
	model = tf.keras.models.Sequential([ 
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(512, activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
	model.fit(x_train, y_train, epochs=5)
	K.clear_session()
	return _
import multiprocessing
ls = [1,2,3]
pool = multiprocessing.Pool()
results = pool.map(train, ls)
pool.close()
pool.terminate()

'''print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save("mnist-tf2.h5")
print("Saved model to disk")'''






