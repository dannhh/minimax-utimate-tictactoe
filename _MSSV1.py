import numpy as np
from numpy.core.numeric import indices
from numpy.random.mtrand import multinomial
from math import inf 
from collections import Counter
from state import UltimateTTT_Move
import copy

def act_move(state, move):
    local_board = state.blocks[move.index_local_board]
    local_board[move.x, move.y] = move.value
    
    state.player_to_move *= -1          
    state.previous_move = move

    return state
    
def select_move(cur_state, remain_time):
    if cur_state.previous_move == None:
        return UltimateTTT_Move(4, 0, 0, 1)
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        best_move = minimax(cur_state, valid_moves, 5)
        return best_move
    return None

def minimax(cur_state, valid_moves, depth):
    global possible_goals
    possible_goals = [([0,0], [1,1], [2,2]), ([0,2], [1,1], [2,0]),
                      ([0,0], [1,0], [2,0]), ([0,1], [1,1], [2,1]), 
                      ([0,2], [1,2], [2,2]), ([0,0], [0,1], [0,2]),
                      ([1,0], [1,1], [1,2]), ([2,0], [2,1], [2,2])]

    best_move = (-inf, None)
    for move in valid_moves:
        state = copy.deepcopy(cur_state)
        state = act_move(state, move)
        value = min_turn(state, depth-1, -inf, inf)
    
        if value > best_move[0]:
            best_move = (value, move)

    return best_move[1]        

def min_turn(cur_state, depth, alpha, beta):
    
    if depth <= 0:
        state = copy.deepcopy(cur_state)
        state.player_to_move *= (-1)
        return evaluate(state)
    
    valid_moves = cur_state.get_valid_moves
    for move in valid_moves:
        state = copy.deepcopy(cur_state)
        state = act_move(state, move)
        value = max_turn(state, depth-1, alpha, beta)

        if value < beta:
            beta = value
        if alpha >= beta:
            break

    return beta

def max_turn(cur_state, depth, alpha, beta):
    if depth <= 0:
        return evaluate(cur_state)
    
    valid_moves = cur_state.get_valid_moves
    for move in valid_moves:
        state = copy.deepcopy(cur_state)
        state = act_move(state, move)
        value = min_turn(state, depth-1, alpha, beta)

        if alpha < value:
            alpha = value
        if alpha >= beta:
            break

    return alpha

def evaluate(cur_state):
    score = 0
    for block_idx in range(9):
        block = cur_state.blocks[block_idx]
        score += evaluate_small_box(cur_state, block)

    return score

def evaluate_small_box(cur_state, block):
    global possible_goals
    score = 0

    player = copy.deepcopy(cur_state.player_to_move)
    three = Counter([player, player, player])
    two   = Counter([player, player, 0])
    one   = Counter([player, 0, 0])

    player = player*(-1)
    three_opponent = Counter([player, player, player])
    two_opponent   = Counter([player, player, 0])
    one_opponent   = Counter([player, 0, 0])

    for idxs in possible_goals:
        (x, y, z) = idxs
        current = Counter([block[x[0]][x[1]], block[y[0]][y[1]]
                           , block[z[0]][z[1]]])

        if current == three:
            score += 100
        elif current == two:
            score += 10
        elif current == one:
            score += 1
        elif current == three_opponent:
            score -= 100
            return score
        elif current == two_opponent:
            score -= 10
        elif current == one_opponent:
            score -= 1

    return score