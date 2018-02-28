# simple classifier example
# note if rows and column of input matrix can be rearrange without much impact den dont use CNN
# CNN work for things that are spatially grouped, specifically finding patterns and classify image

import numpy as np
# reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

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

# conv layer 1 output shape (32, 28, 28)
# conv layer filter away information by sampling image 5x5 pixel at a time simplifying the input dim
# without conv layer just the pass to next layer would be 28*28*3 (if colored)*64
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5, #specify value for all spacial dim
    strides=1, #move the sampling by 1 column
    padding='same',     # Padding method
    data_format='channels_first',
))

model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
# pooling usually after activation, what it does is it shrinks the images by taking the max value in the 2d matrix base on the stride size
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
# fully connect the single layer of 3136 nodes to the 1024 dense layer
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer 0.0004 learning rate
# adam is the combination of momentum and RMSprop
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam, #can also use the default 'adam'
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, Y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, Y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

