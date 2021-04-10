import numpy as np
import random

from task1_environment.environment.character import Character
from task1_environment.policy.baseline import observe

rng = np.random.default_rng()


class A3C_Policy:
    def __init__(self,action):
        self.action = action
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    def policy(self,**policy_kwargs):
        return self.actions[self.action]

