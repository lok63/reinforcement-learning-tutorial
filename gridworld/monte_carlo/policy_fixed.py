from grid_world import standard_grid, print_values, print_policy
import numpy as np
import config

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
    grid.set_state(s) # set the starting state

    # We need to store all states and their according rewards
    # We initialise the list with the first state. It's reward is 0 since this is a starting state
    states_and_rewards = [(s,0)]


    while not grid.game_over():
        a = policy[s]               # get the action for the current state
        r = grid.move(a)            # make a move, now the state has been changed, we also get a reward for the new state
        s = grid.current_state()    # get the new state

        states_and_rewards.append((s,r))


    """
    Calculate the returns 
    
    Now that we have all the rewards stored in a list we start from the last one. 
    We already know that the total return for the last state is 0. So we initialise the last item of our list with the terminal state and G=0
    
    G(t) = r(t+1) + gamma * G(t+1)
    """
    G = 0
    states_and_returns = []
    for s, r in reversed(states_and_rewards):
        states_and_returns.append((s, G))
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
    V = {}
    for state in all_states:
        V[state] = 0

    all_returns = {}

    # Play the game for N times
    for _ in range(N):
        s, g = play_episode(grid, policy)[0]
        seen_states = []
        # for s, g in zip(state, returns):
        # First visit Monte Carlo --> Use the V(s) for the state we saw firs only
        if s not in seen_states:
            if s in all_returns.keys():
                all_returns[s].append(g)
            else:
                all_returns[s] = [g]

            seen_states.append(s)
            V[s] = np.mean(all_returns[s])

    return V





if __name__ =='__main__':

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


    V = first_visit_monte_carlo(grid,policy, N = 100)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
