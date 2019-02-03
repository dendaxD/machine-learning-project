# Inspired by https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, discount_factor = 1.0, render = True):
    state = env.reset()
    total_reward = 0
    step = 0
    while True:
        if render:
            env.render()
        state, reward, done , _ = env.step(int(policy[state]))
        total_reward += (discount_factor ** step * reward)
        step += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, discount_factor = 1.0,  n = 100):
    scores = [
            run_episode(env, policy, discount_factor = discount_factor, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, discount_factor = 1.0):
    policy = np.zeros(state_size)
    for s in range(state_size):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + discount_factor * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, discount_factor = 1.0):
    v = np.zeros(state_size)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(state_size):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(action_size)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped

    state_size = env.observation_space.n
    action_size = env.action_space.n
    discount_factor = 1.0

    optimal_v = value_iteration(env, discount_factor);
    policy = extract_policy(optimal_v, discount_factor)

    policy_score = evaluate_policy(env, policy, discount_factor, n=1000)
    print('Policy average score = ', policy_score)