import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

# This time we load the cifat10 dataset with the load_data method. We also normalize this data immediately after that, by dividing all values by 255.
# Since we are dealing with RGB values, and all values lie in between 0 and 255, we end up with values in between 0 and 1.
# Next, we define the possible class names in a list, so that we can label the final numerical results later on. The neural network will again produce a
# softmax result, which means that we will use the argmax function, to figure out the class name.
class_names = ['Plane', 'Car', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# visualization of the data
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
# For this we run a for loop with 16 iterations and create a 4x4 grid of subplots. The x-ticks and the y-ticks will be set to empty lists, so that we don’t have
# annoying coordinates. After that, we use the imshow method, to visualize the individual images. The label of the image will then be the respective class
# name.

# This dataset contains a lot of images.
train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# building neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
# Here we again define a Sequential model. Our inputs go directly into a convolutional layer (Conv2D ). This layer has 32 filters or channels in the
# shape of 3x3 matrices. The activation function is the ReLU function, which we already know and the input shape is 32x32x3. This is because we our
# images have a resolution of 32x32 pixels and three layers because of the RGB colors. The result is then forwarded into a MaxPooling2D layer that simplifies
# the output. Then the simplified output is again forwarded into the next convolutional layer. After that into another max-pooling layer and into
# another convolutional layer. This result is then being flattened by the Flatten layer, which means that it is transformed into a one-dimensional vector
# format. Then we forward the results into one dense hidden layer before it finally comes to the softmax output layer. There we find the final
# classification probabilities.

# training and testing
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))
# This means that our model is going to see the same data ten times over and over  again.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('loss: ', test_loss)
print('accuracy: {}%'.format( test_acc*100))

# classifying own images
# Since our model is trained, we can now go ahead and use our own images of cars, planes, horses etc. for
# classification.
# If you don’t have your own images, you can use Google to find some.
# The important thing is that we get these images down to 32x32 pixels because this is the required input format of our model. For this you can use any
# software like Gimp or Paint. You can either crop the images or scale them.

# img1 = cv.imread('filename')
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
# img2 = cv.imread('filename')
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# plt.imshow(img1, cmap=plt.cm.binary)
# plt.show()

# The function imread loads the image into our script. Then we use the cvtColor method, in order to change the default color scheme of BGR (blue,
# green, red) to RGB (red, green, blue).

# predict
# prediction = model.predict(np.array([img1]/255))
# index = np.argmax(prediction)
# print(class_names[index])
# First we use the predict function to get the softmax result. Notice that we are converting our image into a NumPy array and dividing it by 255. This is
# because we need to normalize it, since our model was trained on normalized values. Then we use the argmax function to get the index of the highest
# softmax activation value. Finally, we print the class name of that index as a result.
