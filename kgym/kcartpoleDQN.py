import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class Agent():
    def __init__(self, state_size, action_size):
        #back up model
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        #store state action and corresponding reward
        self.memory = deque(maxlen=2000)
        #higher learning rate to converge faster but might miss local maxima
        self.learning_rate = 0.001
        #discount future reward
        self.gamma = 0.95
        #epislon
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        # decay exploration by 0.5% if exploration rate is above 1%
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        #if there is trained model load the weights and reduce exploration
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    # store current state into memory deque
    # note that because we are using deque with maxlen once append exceeds memory size initial memory will be popped
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        #when fresh and no experience
        if len(self.memory) < sample_batch_size:
            return

        #random sampling
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make('CartPole-v1')


        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)
        self.score = []
        self.sum = 0
        self.reward = []
    def run(self):
        try:
            flag = True
            RENDER= False
            #for index_episode in range(self.episodes):
            for index_episode in range(1, 101):



                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                index = 0
                while not done:
                    if RENDER: self.env.render()

                    #                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    if index>200:
                        RENDER = True  # rendering

                print("Episode {}# Score: {}".format(index_episode, index))
                self.agent.replay(self.sample_batch_size)
                self.score.append([index_episode, index + 1])
                self.reward.append(index)
                self.sum+=index+1
                avgscore = self.sum/index_episode
                if avgscore>195 and flag:
                    print("Completed on Episode {}".format(index_episode))

                    flag=False
        finally:
            print("complete")
            #self.agent.save_model()


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
    np_score = np.array(cartpole.score)

    #function to measure score to  episode to measure how well it stacked and compute the final mean score
    # note x axis is episode, y axis is episode score
    plt.plot(np_score[:,0],np_score[:,1])
    plt.show()
    plt.clf()

    plt.plot(len(cartpole.reward), cartpole.reward)
    plt.show()


    #avg score
    print(np.mean(np_score[:,1]))
