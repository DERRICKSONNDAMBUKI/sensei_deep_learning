import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# The mnist class that we import here has the function load_data . This function
# returns two tuples with two elements each. The first tuple contains our
# training data and the second one our test data. In the training data we can find
# 60,000 images and in the test data 10,000.
# The images are stored in the form of NumPy arrays which contain the
# information about the individual pixels
X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)
# we scale our data down.

# Building the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.kersa.layers.Dense(units=10, activation=tf.nn.softmax))
# First we define our model to be a Sequential . This is a linear type of model
# where which is defined layer by layer. Once we have defined the model, we
# can use the add function to add as many different layers as we want.
# The first layer that we are adding here is the input layer. In our case, this is a
# Flatten layer. This type of layer flattens the input. As you can see, we have
# specified the input shape of 28x28 (because of the pixels). What Flatten does
# is to transform this shape into one dimension which would here be 784x1.
# All the other layers are of the class Dense . This type of layer is the basic one
# which is connected to every neuron of the neighboring layers. We always
# specify two parameters here. First, the units parameter which states the
# amount of neurons in this layer and second the activation which specifies
# which activation function we are using.
# We have two hidden layers with 128 neurons each. The activation function
# that we are using here is called relu and stands for rectified linear unit . This
# is a very fast and a very simple function. It basically just returns zero
# whenever our input is negative and the input itself whenever it is positive.

# Because of this we are getting ten different probabilities for each digit, indicating its likelihood.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossntropy', metrics=['accuracy'])
# We are not going to go into the math of the adam optimizer or the sparse_categorical_crossentropy loss function.
# However, these are very popular choices, especially for tasks like this one.

# training and testing
model.fit(X_train, Y_train, epochs=3)
loss, accuracy = model.evaluate(X_test, Y_test)

print('Loss: ', loss)
print('Accuracy', accuracy)

# # predicting your own data
# image = cv.imread('digits.png')[:,:,0]
# image = np.invert(np.array([image]))
# # We use the imread method to read our image into the script. Because of the
# # format we remove the last dimension so that everything matches with the
# # input necessary for the neural network. Also, we need to convert our image
# # into a NumPy array and invert it, since it will confuse our model otherwise.
# prediction = model.prediction(image)
# print('prediction: {}'.format(np.argmax(prediction)))
# # Prediction: 7
# plt.imshow(image[0])
# plt.show()
