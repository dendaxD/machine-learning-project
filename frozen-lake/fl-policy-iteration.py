# Inspired by https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, discount_factor = 1.0, render = True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (discount_factor ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, discount_factor = 1.0, n = 100):
    scores = [run_episode(env, policy, discount_factor, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, discount_factor = 1.0):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + discount_factor * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, discount_factor=1.0):
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + discount_factor * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v

def policy_iteration(env, discount_factor = 1.0):
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    discount_factor = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, discount_factor)
        new_policy = extract_policy(old_policy_v, discount_factor)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped
    optimal_policy = policy_iteration(env, discount_factor = 1.0)
    scores = evaluate_policy(env, optimal_policy, discount_factor = 1.0)
    print('Average scores = ', np.mean(scores))