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

# download keras dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# data preprocessing to 1hot
X_train = X_train.reshape(-1,1,28,28)/255 #first index -1 means batch size, second is channel Mnist is not rgb
X_test = X_test.reshape(-1,1,28,28)/255

# 1hot array for output
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# Build nerual net list
model = Sequential()

# build RNN model

# output layer

# We add metrics to get more results you want to see
adam = Adam(LR)
model.compile(optimizer=adam, #can also use the default 'adam'
              loss='categorical_crossentropy',
              metrics=['accuracy'])