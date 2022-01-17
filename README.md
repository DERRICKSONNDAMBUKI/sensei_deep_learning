# Sensei Deep Learning

Deep learning is a sub-field of machine learning, which is all about neural networks. These neural networks are inspired by the human brain and produce extraordinary results. They beat professional chess players, drive cars and outperform humans even in complex video games like Dota 2.

## 1 – BASICS OF NEURAL NETWORKS

### WHAT ARE NEURAL NETWORKS?

Artificial neural networks are mathematical structures that are inspired by the human brain. They consist of so-called neurons , which are interconnected with each other. The human brain consists of multiple billions of such neurons. Artificial neural networks use a similar principle but also called perceptron.

The structure of a neural network is quite simple. The first one is the input layer and the last one is the output layer. In between we have multiple so-called hidden layers.

### Structure of Neuron

This input is a numerical value and it then gets multiplied by each individual weight (w1, w2, w3...) . At the end we then subtract the bias (b) . The result is the output of that particular connection. These outputs are that forwarded to the next layer of neurons .

### Activation Functions

There are a lot of different so-called activation functions which make everything more complex. These functions determine the output of a neuron. Basically what we do is: We take the input of our neuron and feed the value into an activation function. This function then returns the output value. After that we still have our weights and biases.

### Sigmoid Activation Function

A commonly used and popular activation function is the so-called sigmoid activation function . This function always returns a value between zero and one, no matter what the input is. The smaller the input, the closer the output will be to zero. The greater the input, the closer the output will be to one.

### Relu Actovation Function

The probably most commonly used activation function is the so-called ReLU function . This stands for rectified linear unit . This function is very simple but also very useful. Whenever the input value is negative, it will return zero. Whenever it is positive, the output will just be the input.

### Types of Neural Networks

Neural networks are not only different because of the activation functions of
their individual layers. There are also different types of layers and networks.

#### Feed Forward Neural networks

The connections are pointed into one direction only. The information flows from left to right.

#### Recurrent Neural networks (direct feedlback)

If we take the output of a neuron and use it as an input of the same neuron, we are talking about direct feedback . Connecting the output to neurons of the same layer is called lateral feedback . And if we take the output and feed it into neurons of the previous layer, we are talking about indirect feedback .
The advantage of such a recurrent neural network is that it has a little memory
and doesn’t only take the immediate present data into account. We could say
that it “looks back” a couple of iterations. This kind of neural networks is oftentimes used when the tasks requires the processing of sequential data like text or speech. The feedback is very useful in this kind of tasks. However it is not very useful when dealing with image recognition or image processing.

#### Convolutional Neural Networks

This type is primarily used for processing images and sound. It is especially useful when pattern recognition in noisy data is needed.

#### Training and Testing

In order to make a neural network produce accurate results, we first need to train and test it. For this we use already classified data and split it up into training and testing data. Most of the time we will use 20% of the data for testing and 80% for training. The training data is the data that we use to optimize the performance. The testing data is data that the neural network has never seen before and we use it to verify that our model is accurate.

#### Error and Loss

When evaluating the accuracy or the performance of our model, we use two metrics – error and loss .
Basically you could say that the error indicates how many of the examples were classified incorrectly. This is a relative value and it is expressed in percentages. An error of 0.21 for example would mean that 79% of the examples were classified correctly and 21% incorrectly. This metric is quite easy to understand for humans.
The loss on the other hand is a little bit more complex. Here we use a so-called loss function to determine the value. This value then indicates how bad our model is performing. Depending on the loss function, this value might look quite different. However, this is the value that we want to minimize in order to optimize our model.

#### Gradient Descent

The minimization of this value is done with the help of the so-called gradient descent algorithm .

#### Backpropagation

Basically backpropagation is just the algorithm that calculates the gradient for the gradient descent algorithm. It determines how and how much we need to change which parameters in order to get a better result.
First of all we take the prediction of the model and compare it to the actually desired result.
Let’s say the first neuron is the one that indicates that the picture is a cat. In this case the prediction would say that the picture is a dog (since the second neuron has a higher activation) but the picture is actually one of a cat.
So we look at how the results need to be changed in order to fit the actual data. Notice however that we don’t have any direct influence on the output of the neurons. We can only control the weights and the biases.
In order to change the value of neurons we can either tweak the weights and biases or we can try to change the inputs.

## Summary

- Activation functions determine the activation of a neuron which then influences the outputs.
- The classic neural networks are feed forward neural networks. The information only flows into one direction.
- In recurrent neural networks we work with feedback and it is possible to take the output of future layers as the input of neurons. This creates something like a memory.
- Convolutional neural networks are primarily used for images, audio data and other data which requires pattern recognition. They split the data into features.
- Usually we use 80% of the data we have as training data and 20% as testing data.
- The error indicates how much percent of the data was classified incorrectly.
- The loss is a numerical value which is calculated with a loss function. This is the value that we want to minimize in order to optimize our model.
- For the minimization of the output we use the gradient descent algorithm. It finds the local minimum of a function.
- Backpropagation is the algorithm which calculates the gradient for the gradient descent algorithm. This is done by starting from the output layer and reverse engineering the desired changes.

I highly recommend using a professional environment like PyCharm or Jupyter Notebook.

- PyCharm: https://www.jetbrains.com/pycharm/download/
- Anaconda: https://www.anaconda.com/distribution/
  Tensorflow is the main library here. We will use it to load data sets, build neural networks, train them etc. The other three libraries are not necessary for the functionality of the neural network. We are only using them in order to load our own images of digits at the end.
  Numpy will be used for reformatting our own images and Matplotlib will be used for their visualization.
  CV2 is the OpenCV library and it will allow us to load our images into the script. You will need to install this module separately:

```
$ pip install opencv-python
```

##### Datasets

Keras Datasets: https://keras.io/datasets/
Scikit-Learn Datasets: https://scikit-learn.org/stable/datasets/index.html

e.t.c
YouTube: https://bit.ly/3a5KD2i
Website: https://www.neuralnine.com/
Instagram: https://www.instagram.com/neuralnine/