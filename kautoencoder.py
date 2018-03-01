# simple Simple RNN example

import numpy as np
# reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt


# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Dataset preprocessing
X_train = X_train.astype('float32')/255. -0.5
X_test = X_test.astype('float32')/255. -0.5
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))

# Note encoding only X data
print(X_train.shape)
print(X_test.shape)

# 2 output after compression
encoding_dim = 2

#input place holder #note 28*28 784 pixel image input
input_img = Input(shape=(784,))

# encoder layer
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layer decompress
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded) # cause input consist of -0.5


# construct autoencoder
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=decoded)

# compile auto encoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
print("training---------------")
autoencoder.fit(X_train, X_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)

# plotting
encoded_imgs = encoder.predict(X_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=Y_test)
plt.colorbar()
plt.show()