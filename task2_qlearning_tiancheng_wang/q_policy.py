# https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#naive-q-learning
import numpy as np
from qlearning_tiancheng_wang.q_state import calculate_state
import random

ACTION_SPACE = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def q_policy(state, table, **kwargs):
    qvals = {action: table[state, action] for action in ACTION_SPACE}
    max_q_val = max(qvals.values())
    actions_with_max_q_val = [action for action, qval in qvals.items() if qval == max_q_val]
    return random.choice(actions_with_max_q_val)
