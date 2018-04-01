# First we import some libraries
# Json for loading and saving the model (optional)
import json
# matplotlib for rendering
import matplotlib.pyplot as plt
# numpy for handeling matrix operations
import numpy as np
# time, to, well... keep track of time
import time
# Python image libarary for rendering
from PIL import Image
# iPython display for making sure we can render the frames
from IPython import display
# seaborn for rendering

import seaborn
#Keras is a deep learning libarary
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

#Setup matplotlib so that it runs nicely in iPython
#%matplotlib inline
#setting up seaborn
seaborn.set()


last_frame_time = 0

"""
Class catch is the actual game.
In the game, fruits, represented by white tiles, fall from the top.
The goal is to catch the fruits with a basked (represented by white tiles, this is deep learning, not game design).
"""
class Catch(object):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]





"""
During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory.
In training, batches of randomly drawn experiences are used to generate the input and target for training.
"""
class ExperienceReplay(object):
    """
    Setup
    max_memory: the maximum number of experiences we want to store
    memory: a list of experiences
    discount: the discount factor for future experience

    In the memory the information whether the game ended at the state is stored seperately in a nested array
    [...
    [experience, game_over]
    [experience, game_over]
    ...]
    """
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i+1] = state_t

            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])
            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets






def set_max_fps(last_frame_time, FPS=1):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1. / FPS - (current_milli_time() - last_frame_time) / 1000  # remaining sleep time
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()

def display_screen(action, points, input_t):
    # Function used to render the game screen
    # Get the last rendered frame
    global last_frame_time
    # Render the game with matplotlib
    plt.imshow(input_t.reshape((grid_size,) * 2),
               interpolation='none', cmap='gray')
    # Clear whatever we rendered before
    display.clear_output(wait=True)
    # And display the rendering
    display.display(plt.gcf())
    # Update the last frame time
    last_frame_time = set_max_fps(last_frame_time)





if __name__ == "__main__":
    '''
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 1000
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    grid_size = 10

    # build a three layer NN
    # input is the color of each coordination,  grid_size**2 means the square of gird_size
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size ** 2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Define environment/game
    env = Catch(grid_size)

    # Initialize experience replay object
    # max_memory is the largest state and action list for study
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0  # win count
    # Epochs is the number of games we play
    for e in range(epoch):
        loss = 0.
        env.reset()  # reset the environment, get initial input (clear the state)
        game_over = False

        input_t = env.observe()  # the initial position of fruit and basket

        while not game_over:
            # The learner is acting on the last observed game screen
            # input_t is a vector containing representing the game screen
            input_tm1 = input_t
            # get next action
            
            """
            We want to avoid that the learner settles on a local minimum.
            Imagine you are eating eating in an exotic restaurant. After some experimentation you find 
            that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
            food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
            It's simple: Sometimes, they just don't let you choose but order something random from the menu.
            Maybe you'll like it.
            The chance that your friends order for you is epsilon
            """
            if np.random.rand() <= epsilon:
                # Eat something random from the menu
                action = np.random.randint(0, num_actions, size=1)  # give a random action : left or right or stay
            else:
                # Choose yourself
                # q contains the expected rewards for the actions 
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                action = np.argmax(q[0])  # use model to predict the action

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1
            
            
            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """
            
            # store experience
            # input_tm1 : the  present position of fruit and basket at time t , which is state
            # action : move right or left
            # reward : reward of action
            # input_t : the initial position of fruit and basket at time t+1, which is state
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            # inputs is the state of fruit and basket, targets is the predicted action value, but calibrate by the game result
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            result = model.train_on_batch(inputs, targets)

            # print result
            loss += model.train_on_batch(inputs, targets)

        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    '''

    # Make sure this grid size matches the value used fro training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(10):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        c += 1
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            display_screen(action, c, input_t)
            c += 1
