# Inspired by https://ferdinand-muetsch.de/cartpole-with-qlearning-first-experiences-with-openai-gym.html
# & https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947&

import gym
import numpy as np
import math
from collections import deque


class QLearningCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')

        self.n_episodes = 1000
        self.win_score = 195
        self.min_learning_rate = 0.1
        self.min_exploration_rate = 0.1
        self.discount_factor = 1.0

        self.buckets = (1, 1, 6, 12,) # down-scaling feature space to discrete range
        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, exploration_rate):
        return self.env.action_space.sample() if (np.random.random() <= exploration_rate) else np.argmax(self.q_table[state])

    def update_q(self, old_state, action, reward, new_state, learning_rate):
        self.q_table[old_state][action] += learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state]) - self.q_table[old_state][action])

    def get_exploration_rate(self, t):
        return max(self.min_exploration_rate, min(1, 1.0 - math.log10((t + 1) / 25)))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def run(self):
        scores = deque(maxlen=100)

        for episode in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())
            learning_rate = self.get_learning_rate(episode)
            exploration_rate = self.get_exploration_rate(episode)
            done = False
            score = 0

            while not done:
                action = self.choose_action(current_state, exploration_rate)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, learning_rate)
                current_state = new_state
                score += reward

        # Pritout of results
            scores.append(score)
            mean_score = np.mean(scores)
            if mean_score >= self.win_score and episode >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(episode, episode - 100))
                return episode - 100
            if episode % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(episode, mean_score))

        print('Did not solve after {} episodes 😞'.format(episode))
        return episode


if __name__ == "__main__":
    solver = QLearningCartPoleSolver()
    solver.run()