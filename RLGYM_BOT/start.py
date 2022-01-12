import rlgym
import random
import numpy as np
from AirStateSetter import InitialStateSetter
from Reward import AirReward
#from rlgym.utils.reward_functions.common_rewards import ConditionalRewardFunction
from rlgym.utils.reward_functions import RewardFunction
from TouchArenaBoundariesTerminalCondition import CollidedTerminalCondition
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import activations

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 110

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

env = rlgym.make(game_speed=100, state_setter=InitialStateSetter(), reward_fn = AirReward(), terminal_conditions=[CollidedTerminalCondition(), TimeoutCondition(max_steps)])

class Memory():
    """
    Class that serves as a medium to persist the actions of the agent in the training process (used for replay memory)
    """
    def __init__(self, buffer_size = 1_000_000) -> None:
        self.buffer_size = buffer_size
        self.buffer = []
        self.idx = 0
        self.total_elements = 0

    def add(self, x):
        self.buffer.append(None)
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.buffer_size

    def size(self):
        return len(self.buffer)

def build_model():
    """
    Method that builds a predefined model consisting of:
    -an input layer with 70 nodes: RelU activation;
    -a hidden layer with 100 nodes: tanh activation -> values between -1 and 1;
    -an output layer with 8 nodes.
    """
    model = Sequential()
    model.add(Dense(100, input_shape=(70,), activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

def deep_copy_model(model):
    """
    Method that takes a neural network as input and creates and returns a deep copy of it (used for replay memory)
    """
    copied_model = keras.models.clone_model(model)
    copied_model.build((None, 70))
    copied_model.compile(optimizer='adam', loss='mean_squared_error')
    copied_model.set_weights(model.get_weights())
    return copied_model

# deepq learning models; two are used in order to implement the replay memory functionality
qnn = build_model()
tnn = deep_copy_model(qnn)

# replay memory
memory = Memory()

# number of entries sampled from the replay memory to learn at a time
batch_size = 32

# used in epsilon-greedy action-making
epsilon = 1.0

# discount factor
gamma = 0.8

# number of played games & frames per game
num_episodes = 1500
frames_per_episode = 64

for e in range(num_episodes):
    
    obs = env.reset()

    for frame in range(frames_per_episode):
        print(f'frame {frame}, episode {e}')
        memory.total_elements += 1
        if memory.total_elements < 5000:
            # random action if no more than 5000 moves have been visited
            action = env.action_space.sample()
        else:
            if random.random() < epsilon:
                # epsilon-greedy action
                action = env.action_space.sample()
            else:
                # qnn prediction
                curr_state_tensor = np.array(obs).reshape((70, 1)).T
                prediction_tensor = qnn.predict(curr_state_tensor)[0]
                action = prediction_tensor
            epsilon *= 0.95

        action[4] = -1.0
        next_obs, reward, done, gameinfo = env.step(action)

        # persist the tuple in the replay
        memory.add((obs, action, reward, next_obs, done))

        if frame % 16 == 0 and frame != 0 and memory.size() >= batch_size:
            batch = random.sample(memory.buffer, batch_size)
            features = []
            labels = []

            inputs = []
            for i in range(len(batch)):
                inputs.append(batch[i][0])

            inputs = np.array(inputs).reshape((batch_size, 70))
            outputs_qnn = qnn.predict(inputs)

            inputs = []
            for i in range(len(batch)):
                inputs.append(batch[i][3])

            inputs = np.array(inputs).reshape((batch_size, 70))
            outputs_tnn = tnn.predict(inputs)

            for idx, (_obs, _action, _reward, _next_obs, _done) in enumerate(batch):
                curr_state_tensor = np.array(_obs).reshape((70, 1)).T
                next_state_tensor = np.array(_next_obs).reshape((70, 1)).T

                if _done:
                    # no need for further predictions, game has ended already, take the reward for granted
                    target = _reward
                else:
                    # game hasn't ended yet, take the reward and make a prediction to update the target
                    target = _reward + gamma * np.max(outputs_tnn[idx])

                target_tensor = outputs_qnn[idx].reshape((8, 1)).T
                target_tensor[0][np.argmax(_action)] = target
                features.append(curr_state_tensor)
                labels.append(target_tensor)
            
            features = np.array(features).reshape((batch_size, 70))
            labels = np.array(labels).reshape((batch_size, 8))
            qnn.fit(features, labels, verbose=0)

        # update the current state
        obs = next_obs

        if done:
            break
    
    # copy weight to target neural net after episode finishes
    tnn = deep_copy_model(qnn)

qnn.save('my_model')