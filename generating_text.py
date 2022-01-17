import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM
# The library random will be used later on in a helper function. We have already used numpy . We used the basic tensorflow library was used to load
# the data from the internet. For building our neural network, we will once again need the Sequential
# model from Keras. This time however, we will use a different optimizer, namely the RMSprop . And of course we also import the layer types, which
# we are going to use.

# download file
filepath = tf.keras.utils.get_file(
    # 'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb')\
    .read().decode(encoding='utf-8').lower()
# We use the get_file method of keras.utils in order to download the file. The first parameter is the filename that we choose and the second one is the link.
# After that we open the file, decode it and save it into a variable. Notice that we are using the lower function at the end. We do this because it drastically
# increases the performance, since we have much less possible characters to choose from. And for the semantics of the text the case is irrelevant.

# preparing data
# The problem that we have right now with our data is that we are dealing with text. We cannot just train a neural network on letters or sentences. We need to
# convert all of these values into numerical data. So we have to come up with a system that allows us to convert the text into numbers, to then predict specific
# numbers based on that data and then again convert the resulting numbers back into text.
text = text[300000:800000]  # number of characters

# converting text
# We are not going to use the ASCII codes but our own indices.
characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# splitting text
SEQ_LENGTH = 40
STEP_SIZE = 3
sentences = []
next_char = []

for i in range(0, len(text)-SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_char.append(text[i+SEQ_LENGTH])

# Here we run a for loop and iterate over our text with the given sequence length and step size. The control variable i gets increased by STEP_SIZE with each iteration.

# converting to numpy format
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)
# For this we first create two NumPy arrays full of zeroes. These zeroes however are actually False values, because our data type is bool which stands
# for Boolean . The x array is three-dimensional and it shapes is based on the amount of sentences, the length of those and the amount of possible
# characters. In this array we store the information about which character appears at which position in which sentence. Wherever a character occurs, we
# will set the respective value to one or True .

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
        y[i, char_to_index[next_char[i]]] = 1
# We use the enumerate function two times so that we know which indices we need to mark with a
# one. Here we use our char​ _to_index dictionary, in order to get the right index for each character.
# To make all of that a little bit more clear, let us look at an example. Let’s say the character ‘g’ has gotten the index 17. If this character now occurs in the
# third sentence (which means index two), at the fourth positon (which means index three), we would set x[2,3,17] to one.

# build recurrent neural network
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
# Our model is actually quite simple. The inputs go directly into an LSTM layer with 128 neurons. We define the input shape to be the sequence length times
# 320the amount of possible characters. We already talked about how this layer works. This layer is the memory of our model. It is then followed by a Dense
# layer with as many neurons as we have possible characters. This is our hidden layer. That adds complexity and abstraction to our network. And then last but
# not least we have the output layer, which is an Activation layer. In this case it once again uses the softmax function that we know from the last chapter.
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=1)
# Now we compile our model and optimize it. We choose a learning rate of 0.01. After that we fit our model on the training data that we prepared. Here
# we choose a batch_size of 256 and four epochs . The batch size indicates how many sentences we are going to show the model at once.

# helper function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds= exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# This function will later on take the predictions of our model as a parameter and then choose a “next character”. The second parameter temperature
# indicates how risky or how unusual the pick shall be. A low value will cause a conservative pick, whereas a high value will cause a more experimental pick.
# We will use this helper function in our final function.

# generating texts


def generate_text(length, temperature):
    start_index = random.randint(0, len(text)-SEQ_LENGTH-1)
    generated = ''
    sentence = text[start_index:start_index+SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1
            predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:]+next_character
    return generated
# We then convert this initial text again into a NumPy array. After that we feed these x-values into our neural network and predict the output. For this we use
# the predict method. This will output the probabilities for the next characters. We then take these predictions and pass them to our helper function. You
# have probably noticed that we also have a temperature parameter in this function. We directly pass that to the helper function.
# In the end we receive a choice from the sample function in numerical format.
# This choice needs to be converted into a readable character, using our second 322dictionary. Then we add this character to our generated text and repeat this
# process until we reach the desired length.


# results
print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))
