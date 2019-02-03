import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Initialise environment
env = gym.make('FrozenLake-v0')

state_size = 1
action_size = env.action_space.n

n_episodes = 1000

batch_size = 32 # size of the sample (created from replay memory) NN retrains at after each episode


#Define agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end

        self.discount_rate = 0.9

        self.exploration_rate = 1.0
        self.exploration_decay_rate = 0.001
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01

        self.learning_rate = 0.2 # rate at which NN adjusts models parameters via SGD to reduce cost

        self.model = self._build_model() # private method 
    

    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu')) # states as input
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # 4 actions, so 4 output neurons
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later (replay memory)

    def act(self, state):
        # Exploration vs exploitation +choice of action
        if np.random.rand() <= self.exploration_rate: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch:
            target = reward 
            if not done: 
                target = (reward + self.discount_rate *\
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat

        # Exploration rate decay 
        self.exploration_rate = self.min_exploration_rate +\
            (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-self.exploration_decay_rate*e)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Initialise agent
agent = DQNAgent(state_size, action_size)

# Interact with environment
done = False
for e in range(n_episodes): # iterate over new episodes of the game
    state = env.reset() # reset state at start of each new episode of the game
    state = np.reshape(state, [1,1])
    
    for time in range(5000):

        # env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1,1])
        agent.remember(state, action, reward, next_state, done) 
        state = next_state

        if done: # episode ends if agent drops pole or we reach timestep 5000
            print("episode: {}/{}, score: {}, time: {}, e: {:.2}" # print the episode's score and agent's epsilon
                  .format(e, n_episodes, reward, time, agent.exploration_rate))
            break # exit loop

    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode

env.close()