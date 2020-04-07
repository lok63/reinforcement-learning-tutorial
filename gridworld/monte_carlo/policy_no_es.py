from grid_world import standard_grid, negative_grid, print_values, print_policy
import numpy as np
import config
from tqdm import tqdm
import matplotlib.pyplot as plt


def random_action(a, eps = 0.1):
  # choose given a with probability 1 - eps + eps/4
  # choose some other a' != a with probability eps/4
  p = np.random.random()
  # if p < (1 - eps + eps/len(ALL_POSSIBLE_ACTIONS)):
  #   return a
  # else:
  #   tmp = list(ALL_POSSIBLE_ACTIONS)
  #   tmp.remove(a)
  #   return np.random.choice(tmp)
  #
  # this is equivalent to the above
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(config.ALL_POSSIBLE_ACTIONS)


def play_episode(grid, policy):
    """
    Function that start an episode of GridWorld. The goal of this function is to calculate returns and rewards
    :param policy: a dictionary that shows all the actions for each state
    :return: states_and_returns : list of tuples e.g states_and_returns[(s,G)] where s is a state (0,1) and G is the return variable
    """

    # We always start from the same state
    s = (2,0)
    grid.set_state(s) # set the starting state

    #Get a random acton
    a = np.random.choice(config.ALL_POSSIBLE_ACTIONS)

    # We need to store all states and their according rewards
    # We initialise the list with the first state. It's reward is 0 since this is a starting state
    states_actions_rewards = [(s, a, 0)]

    while True:
        r = grid.move(a)            # make a move, now the state has been changed, we also get a reward for the new state
        s = grid.current_state()    # get the new state

        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])  # the next state is stochastic
            states_actions_rewards.append((s, a, r))

    """
    Calculate the returns 
    
    Now that we have all the rewards stored in a list we start from the last one. 
    We already know that the total return for the last state is 0. So we initialise the last item of our list with the terminal state and G=0
    
    G(t) = r(t+1) + gamma * G(t+1)
    """
    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + config.GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    return states_actions_returns

    # print("@@@@@@@")
    # print(list(reversed(states_and_returns)))
    return list(reversed(states_actions_returns))

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

def first_visit_monte_carlo(grid, policy, N):
    """
    :param policy: a dictionary that shows all the actions for each state
    :param N: number of times we wish to play the game
    :return: V : a dictionary with the value function for each state
    """

    # initialize Q(s,a) and returns
    Q = {}
    all_returns = {}  # dictionary of state -> list of returns we've received
    all_states = grid.all_states()
    for s in all_states:
        if s in grid.actions:  # not a terminal state
            Q[s] = {}
            for a in config.ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0  # needs to be initialized to something so we can argmax it
                all_returns[(s, a)] = []
        else:
            # terminal state or state we can't otherwise get to
            pass

    # repeat until convergence
    deltas = []
    for t in range(config.ITERATIONS):

        # generate an episode using pi
        biggest_change = 0
        states_actions_returns = play_episode(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                all_returns[sa].append(G)
                Q[s][a] = np.mean(all_returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    return Q





if __name__ =='__main__':

    grid = negative_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)


    policy = {
        (0, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (0, 1): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (0, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (1, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (1, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 1): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 3): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
    }


    Q = first_visit_monte_carlo(grid,policy, N = 100)



    # find the optimal state-value function
    # V(s) = max[a]{ Q(s,a) }
    V = {}
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1]

    print("final values:")
    print_values(V, grid)
    print("final policy:")
    print_policy(policy, grid)