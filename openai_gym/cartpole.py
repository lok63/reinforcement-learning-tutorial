import gym

"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """


import numpy as np
from tqdm import tqdm

env = gym.make('CartPole-v0')
box = env.observation_space

"""  Random search """


def play_episode(weights):
    env.render()
    # Start on the start state
    env.reset()

    done = False
    n_steps = 0

    # Get a random action for starters
    action = env.action_space.sample()

    while not done:
        observation, reward, done, info = env.step(action)

        dot_product = np.dot(observation, weights)

        action = 1 if dot_product > 1 else 0
        n_steps +=1

    return n_steps


n_episodes = 100
for _ in tqdm(range(100)):
    new_weights = np.random.random(box.shape)
    avg_steps_per_game = []
    best = 0
    # Play the game and count the steps until completion
    for j in range(n_episodes):
        n_steps = play_episode(weights)
        avg_steps_per_game.append(n_steps)

    if sum(avg_steps_per_game) / n_episodes >= best:
        weights = new_weights
        best = sum(avg_steps_per_game) / n_episodes


print(best)
