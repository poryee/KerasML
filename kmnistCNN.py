#simple classifier example

import numpy as np
#reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

#download keras dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#data preprocessing to 1hot
X_train = X_train.reshape(X_train.shape[0],-1)/255 #normalise reshape row and unknown -1 column
X_test = X_test.reshape(X_test.shape[0],-1)/255
#1hot array for output
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

#Build nerual net list
model = Sequential([
    #28*28 =784 pixels
    #32 output 784 input for the very first layer
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    #squeeze all value within list to spread as probability of 0 and 1 according to their value from their sum
    Activation('softmax')

])

#optimisation to converge faster
#epsilon so that division is not 0 think of it as bias recommended 10^-8
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(
    #rmsprop used to dampen oscillation to
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],


)

#training another way
print("training---------------")
model.fit(X_train,Y_train,nb_epoch=2,batch_size=32)

#test
print("\ntest------------------")
loss, accuracy = model.evaluate(X_test, Y_test, batch_size=40)
print("loss:", loss)
print("accuracy:", accuracy)
