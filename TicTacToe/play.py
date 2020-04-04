from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from agent import Agent
from environment import Environment
from human import Human
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

LENGTH = 3

# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try -1, 0, 1)
# impossible games are ignored, i.e. 3x's and 3o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i, j] = v  # if empty board it should already be 0
        if j == 2:
            # j goes back to 0, increase i, unless i = 2, then we are done
            if i == 2:
                # the board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            # increment j, i stays the same
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


# play all possible games
# need to also store if game is over or not
# because we are going to initialize those values to 0.5
# NOTE: THIS IS SLOW because MANY possible games lead to the same outcome / state
# def get_state_hash_and_winner(env, turn='x'):
#   results = []

#   state = env.get_state()
#   # board_before = env.board.copy()
#   ended = env.game_over(force_recalculate=True)
#   winner = env.winner
#   results.append((state, winner, ended))

#   # DEBUG
#   # if ended:
#   #   if winner is not None and env.win_type.startswith('col'):
#   #     env.draw_board()
#   #     print "Winner:", 'x' if winner == -1 else 'o', env.win_type
#   #     print "\n\n"
#   #     assert(np.all(board_before == env.board))

#   if not ended:
#     if turn == 'x':
#       sym = env.x
#       next_sym = 'o'
#     else:
#       sym = env.o
#       next_sym = 'x'

#     for i in xrange(LENGTH):
#       for j in xrange(LENGTH):
#         if env.is_empty(i, j):
#           env.board[i,j] = sym
#           results += get_state_hash_and_winner(env, next_sym)
#           env.board[i,j] = 0 # reset it
#   return results


def initialV_x(env, state_winner_triples):
    # initialize state values as follows
    # if x wins, V(s) = 1
    # if x loses or draw, V(s) = 0
    # otherwise, V(s) = 0.5
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def initialV_o(env, state_winner_triples):
    # this is (almost) the opposite of initial V for player x
    # since everywhere where x wins (1), o loses (0)
    # but a draw is still 0 for o
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V


def play_game(p1, p2, env, draw=False):
    # loops until the game is over
    current_player = None
    while not env.game_over():
        # alternate between players
        # p1 always starts first
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        # draw the board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()

        # current player makes a move
        current_player.take_action(env)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    # do the value function update
    p1.update(env)
    p2.update(env)


if __name__ == '__main__':
    # train the agent
    p1 = Agent()
    p2 = Agent()

    # set initial V for p1 and p2
    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10
    for t in tqdm(range(T)):
        # if t % 200 == 0:
        #     print(t)
        play_game(p1, p2, Environment())

    # play human vs. agent
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        # I made the agent player 1 because I wanted to see if it would
        # select the center as its starting move. If you want the agent
        # to go second you can switch the human and AI.
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break



# import numpy as np
# d = np.zeros([3,3])
# d[0,0] = 1
# d[1,1] = -1
#
#
# def get_state(board = d):
#     LENGTH = 3
#
#     h = 0  # variable to accumulate the final result
#     k = 0  # the power e.g (3^k)
#
#     for i in range(LENGTH):
#         for j in range(LENGTH):
#             # The location is empty
#             if board[i, j] == 0:
#                 v = 0
#             # The locations is marked with x
#             elif board[i, j] == 1:
#                 v = 1
#             # The locations is marked with o
#             elif board[i, j] == -1:
#                 v = 2
#
#             h += (3 ** k) * v
#             k += 1
#
#     return h