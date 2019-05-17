#Imports
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras
import numpy as np
import matplotlib  
matplotlib.use('TkAgg') #Mac matplot workaround   
import matplotlib.pyplot as plt  


#Functions
def plot_img(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap=plt.cm.binary)
	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'green'
	else:
		color = 'red'
	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('green')



#Load data
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#Map imgs to classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Preprocessing

#Visualization of pixel values
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()
#screenshots/Figure_1.png

#Scale images from 0-1 to feed into neural net
train_images = train_images / 255.0
test_images = test_images / 255.0


#Set up layers
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)), #Flatten makes a (784,) input array
	keras.layers.Dense(128, activation=tf.nn.relu), #By flattening the input the dense layers output will be (128) instead of (28, 128)
	keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training
model.fit(train_images, train_labels, epochs=10) 

#Output:
#Epoch 1/10 60000/60000 [==============================] - 29s 488us/sample - loss: 0.4945 - acc: 0.8269
#Epoch 2/10 60000/60000 [==============================] - 29s 478us/sample - loss: 0.3725 - acc: 0.8660
#Epoch 3/10 60000/60000 [==============================] - 32s 540us/sample - loss: 0.3371 - acc: 0.8761
#Epoch 4/10 60000/60000 [==============================] - 30s 499us/sample - loss: 0.3145 - acc: 0.8836
#Epoch 5/10 60000/60000 [==============================] - 30s 505us/sample - loss: 0.2968 - acc: 0.8911
#Epoch 6/10 60000/60000 [==============================] - 30s 507us/sample - loss: 0.2824 - acc: 0.8939
#Epoch 7/10 60000/60000 [==============================] - 30s 503us/sample - loss: 0.2704 - acc: 0.8992
#Epoch 8/10 60000/60000 [==============================] - 33s 543us/sample - loss: 0.2592 - acc: 0.9037
#Epoch 9/10 60000/60000 [==============================] - 33s 555us/sample - loss: 0.2492 - acc: 0.9072
#Epoch 10/10 60000/60000 [==============================] - 34s 565us/sample - loss: 0.2419 - acc: 0.9092

#Accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
#Output:
#Test accuracy: 0.8794

#Predictions
predictions = model.predict(test_images)
predictions[0]

#Prediction visualization
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_img(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
# plt.show()
#screenshots/Figure_2.png

#Single image prediction
img = test_images[0]

#Batch where image is only memeber
img = (np.expand_dims(img, 0))

single_prediction = model.predict(img)
print(single_prediction)
#Output:
#[[4.0423952e-06 1.7867489e-08 1.1013570e-05 8.0041261e-08 3.1214680e-07
#	1.8548490e-03 1.4891983e-05 1.1522283e-02 2.6472981e-06 9.8658991e-01]]


plot_value_array(0, single_prediction, test_labels)
plt.xticks(range(10), class_names, rotation=45)
# plt.show()
#screenshots/Figure_3.png

#Prediction result
result = np.argmax(single_prediction[0])
print(result)
#Output:
#9
