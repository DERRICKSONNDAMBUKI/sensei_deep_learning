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
# Sensei Deep Learning
