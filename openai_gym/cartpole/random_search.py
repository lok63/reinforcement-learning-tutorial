import gym
import numpy as np
from gym import wrappers
from tqdm import tqdm

"""
1. Play the game by initialising random weights each time
2. Policy is the weighted sum of the weights with the observation and take actions if the result is >0 or <0
"""

# TODO: dot_product > 1 yields better results
def get_action(s,w):
    dot_product = np.dot(s, w)
    return 1 if dot_product > 0 else 0


def play_episode(env, weights, render = False):
    if render: env.render()

    # Start on the start state
    observation = env.reset()
    done = False
    n_steps = 0
    rewards = []

    while not done:
        action = get_action(observation,weights)
        n_steps +=1
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if done: break # If done then don't increment the number of steps


    return n_steps , sum(rewards) / n_steps

def play_multiple_episodes(env,new_weights, n_episodes = 100):
    episode_lengths = []

    for _ in range(n_episodes):
        n_steps, rewards = play_episode(env, new_weights)
        episode_lengths.append(n_steps)

    avg_length = sum(episode_lengths) / len(episode_lengths)
    return avg_length

def random_search(env, epochs = 100):
    best = 0
    best_weights = None
    for i in range(epochs):
        new_weights = np.random.random(box.shape) * 2 - 1
        # Play the game with random weights and count the steps until completion
        mean_steps = play_multiple_episodes(env, new_weights)
        print(f'Round: {i+1}: Number of steps: {mean_steps}')

        if mean_steps > best:
            best_weights = new_weights
            best = mean_steps

    print(best)
    return best_weights


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    box = env.observation_space

    best_weights = random_search(env)

    env = wrappers.Monitor(env, '/Users/leo/GymVideos', force= True)
    print(play_episode(env, best_weights, render=True))
