# simple Simple RNN example

import numpy as np
# reproducibility
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import Adam

# Variables
BATCH_INDEX = 0
TIMES_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20 # how many hidden nodes that act as memory cell
LR = 0.006

def get_batch():
    global BATCH_INDEX,TIMES_STEPS

    xs = np.arange(BATCH_INDEX, BATCH_INDEX+TIMES_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIMES_STEPS))/(10*np.pi)
    # sin of x
    seq = np.sin(xs)
    # cos of x
    res = np.cos(xs)

    BATCH_INDEX +=  TIMES_STEPS
    #plt.plot(xs[0,:], res[0,:], seq[0,:], 'b--')
    #plt.show()
    # return seq and res with added column vector at Y
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis], xs]


#get_batch()

# Build RNN model
model = Sequential()

# build LSTM RNN
model.add(LSTM(

    batch_input_shape=(BATCH_SIZE, TIMES_STEPS, INPUT_SIZE),
    output_dim = CELL_SIZE,
    # default is false only output at last time step
    # however when True model return output each time step
    return_sequences=True,
    # true if batch is related to next batch
    stateful = True
))

# output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

# We add metrics to get more results you want to see
adam = Adam(LR)
model.compile(optimizer=adam, #can also use the default 'adam' with the quotes but cannot adjust learning rate
              loss='mse',)

# training
print("training---------------")
for step in range (4001):
    # batch processing slicing from X_Train and Y_Train
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)

    plt.plot(xs[0,:], Y_batch[0].flatten(), 'r', xs[0,:], pred.flatten()[:TIMES_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.5)

    if step %20 == 0:
        print('train cost: ', cost)
