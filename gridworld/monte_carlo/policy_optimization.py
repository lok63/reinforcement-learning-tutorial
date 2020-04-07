from grid_world import standard_grid, print_values, print_policy
import numpy as np
import config
from tqdm import tqdm


def random_action(a):
  # choose given a with probability 0.5
  # choose some other a' != a with probability 0.5/3
  p = np.random.random()
  if p < 0.5:
    return a
  else:
    tmp = list(config.ALL_POSSIBLE_ACTIONS)
    tmp.remove(a)
    return np.random.choice(tmp)


def play_episode(grid, policy):
    """
    Function that start an episode of GridWorld. The goal of this function is to calculate returns and rewards
    :param policy: a dictionary that shows all the actions for each state
    :return: states_and_returns : list of tuples e.g states_and_returns[(s,G)] where s is a state (0,1) and G is the return variable
    """

    # We have to start playing the game from some random position
    all_states = grid.all_states()
    random_choice = np.random.choice(len(all_states))
    s = list(all_states)[random_choice] # Starting state

    while grid.is_terminal(s):
        random_choice = np.random.choice(len(all_states))
        s = list(all_states)[random_choice]  # Starting state


    print(f'Current state: {s}')
    grid.set_state(s) # set the starting state

    # We need to store all states and their according rewards
    # We initialise the list with the first state. It's reward is 0 since this is a starting state
    states_and_rewards = [(s,0, policy[s])]

    while not grid.game_over():
        a = policy[s]               # get the action for the current state
        r = grid.move(a)            # make a move, now the state has been changed, we also get a reward for the new state
        s = grid.current_state()    # get the new state

        states_and_rewards.append((s,r,a))


    """
    Calculate the returns 
    
    Now that we have all the rewards stored in a list we start from the last one. 
    We already know that the total return for the last state is 0. So we initialise the last item of our list with the terminal state and G=0
    
    G(t) = r(t+1) + gamma * G(t+1)
    """
    G = 0
    states_and_returns = []
    for s, r, a in reversed(states_and_rewards):
        states_and_returns.append((s, G, a))
        G = r + config.GAMMA * G

    # print("@@@@@@@")
    # print(list(reversed(states_and_returns)))
    return list(reversed(states_and_returns))


def first_visit_monte_carlo(grid, policy, N):
    """
    :param policy: a dictionary that shows all the actions for each state
    :param N: number of times we wish to play the game
    :return: V : a dictionary with the value function for each state
    """

    # Initialise all value functions with 0 or with a random value
    all_states = grid.all_states()
    Q = {}
    for state in all_states:
        Q[state] = 0

    all_returns = {}

    # Play the game for N times
    for i in tqdm(range(config.ITERATIONS)):
        print(i)
        s, g, a = play_episode(grid, policy)[0]
        print("finished episode")
        seen_states = []
        # for s, g in zip(state, returns):
        # First visit Monte Carlo --> Use the V(s) for the state we saw firs only
        if s not in seen_states:
            if s in all_returns.keys():
                all_returns[(s,a)].append(g)
            else:
                all_returns[(s,a)] = [g]

            seen_states.append(s)
            Q[(s,a)] = np.mean(all_returns[(s,a)])


    return Q





if __name__ =='__main__':

    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)


    policy = {
        (2, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (1, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (0, 0): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (0, 1): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (0, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (1, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 1): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 2): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
        (2, 3): np.random.choice(config.ALL_POSSIBLE_ACTIONS),
    }


    V = first_visit_monte_carlo(grid,policy, N = 100)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
