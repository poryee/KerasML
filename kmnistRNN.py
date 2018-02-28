# simple Simple RNN example

import numpy as np
# reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam

# Variables
TIMES_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50 # how many hidden nodes that act as memory cell
LR = 0.001

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# data preprocessing to 1hot
X_train = X_train.reshape(-1,28,28)/255 #first index -1 means batch size
X_test = X_test.reshape(-1,28,28)/255

# 1hot array for output
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# Build RNN model
model = Sequential()

# build RNN Cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIMES_STEPS, INPUT_SIZE),
    output_dim = CELL_SIZE,
    unroll=True
))

# output layer
# Fully connected final layer to output size (10) for 10 classes
model.add(Dense(OUTPUT_SIZE))
#default softmax is tanh
model.add(Activation('softmax'))

# We add metrics to get more results you want to see
adam = Adam(LR)
model.compile(optimizer=adam, #can also use the default 'adam'
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range (4001):
    # batch processing slicing from X_Train and Y_Train
    X_batch = X_train[BATCH_INDEX: BATCH_SIZE+BATCH_INDEX, :, :]
    # note that Y is 2D hence lesser ':' till the end slice
    Y_batch = Y_train[BATCH_INDEX: BATCH_SIZE + BATCH_INDEX, :]
    cost = model.train_on_batch(X_batch, Y_batch)

    # memory pointer for where to slice the array
    BATCH_INDEX += BATCH_SIZE
    # reset pointer if exceeds array size
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    # for every 500 step in training up to 4k
    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
