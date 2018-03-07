#build simple model to mimic linear regression function

import numpy as np
#reproducibility
np.random.seed(3)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model

# create line data with 200 x value spread between -1 to 1
X = np.linspace(-1,1,200)
# randomise the order
np.random.shuffle(X)

# end result we want model to have weight as close to 0.5 and bias 2
Y = 0.5 * X + 2 + np.random.normal(0, 0.05,(200,)) #add some noise between 0 and 0.05 to bias
# plot data
#plt.scatter(X,Y)
#plt.show()

# first 160 data point
X_train, Y_train = X[:160], Y[:160]
# last 40 to test
X_test, Y_test = X[160:], Y[160:]


# build NN model using keras
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))
# automatically match previous output to next layer input
# model.add(Dense(output_dim=1,))

# chose loss function and optimisation method
# minimal squared error and stochastic gradient decent
model.compile(loss='mse',optimizer='sgd')

# training
print("training---------------")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print ('train cost ', cost)


# save
print("\ntest before save: ", model.predict(X_test[0:2]))
# HDF5, pip3 install h5py
model.save('my_model.h5')
del model

# load
model = load_model('my_model.h5')
print("\ntest after load: ", model.predict(X_test[0:2]))

'''
# notes
# only save the weights while keeping the structure
model.save_weights('asd.h5')
model.load_weights('asd.h5')



# test
print("\ntest------------------")
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("cost:", cost)
# first dense layer
W, b = model.layers[0].get_weights()
print("Weights=", W, "\nbiases=",b)

# plot the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
'''