import numpy as np
import gym
import random
import time


def choose_action(state):
    exploration_rate_treshold = random.uniform(0,1)
    if exploration_rate_treshold > exploration_rate:
        return np.argmax(q_table[state,:])
    else:
        return env.action_space.sample()

#Initialise the environment
env = gym.make("FrozenLake-v0")

state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

#Initialize parameters
n_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


#Tracking of results
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(n_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        # Take new action
        action = choose_action(state)
        new_state, reward, done, info = env.step(action)

        # What would be next action?
        next_action = choose_action(new_state)

        # Update Q-table
        q_table[state, action] = q_table[state, action]*(1 - learning_rate) +\
        	learning_rate*(reward + discount_rate*q_table[new_state,next_action])

        state = new_state
        rewards_current_episode += reward

        # if done:
        #     print("episode: {}/{}, score: {}, step: {}, e: {:.2}" # print the episode's score and agent's epsilon
        #           .format(episode, n_episodes, reward, step, float(exploration_rate)))
        #     break

    # Exploration rate decay 
    exploration_rate = min_exploration_rate +\
    	(max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),n_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# # Print updated Q-table
# print("\n\n********Q-table********\n")
# print(q_table)
        
# env.close()