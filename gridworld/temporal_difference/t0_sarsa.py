from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid ,print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def random_action(a, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
  # returns a list of states and corresponding rewards (not returns as in MC)
  # start at the designated start state


  # initialize Q(s,a) and returns
  Q = {}
  all_states = grid.all_states()

  for s in all_states:
    if s in grid.actions:  # not a terminal state
      Q[s] = {}
      for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0  # needs to be initialized to something so we can argmax it
    else:
      # terminal state or state we can't otherwise get to
      pass

  # let's also keep track of how many times Q[s] has been updated
  update_counts = {}
  update_counts_sa = {}
  for s in all_states:
    update_counts_sa[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      update_counts_sa[s][a] = 1.0



  s = (2, 0)
  grid.set_state(s)
  states_and_rewards = [(s, 0)] # list of tuples of (state, reward)
  a = policy[s]

  while not grid.game_over():
    a = random_action(a)

    r = grid.move(a)
    s_next = grid.current_state()
    a_next = random_action(policy[s_next])

    Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * Q[s_next][a_next] - Q[s][a])

    s = s_next
    a = a_next
  return Q


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }



  # repeat until convergence
  for it in range(1000):
    # generate an episode using pi
    Q = play_game(grid, policy)


  print("values:")
  print_values(Q, grid)
  print("policy:")
  print_policy(policy, grid)