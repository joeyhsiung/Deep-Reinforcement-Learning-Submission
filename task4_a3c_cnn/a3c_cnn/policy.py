import torch
import torch.nn.functional as F
from a3ctwo.utils import v_wrap, set_init, push_and_pull, record

ACTION_MAP = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1)
}


class A3CTwo:
    def __init__(self, model):
        self.action_map = ACTION_MAP
        self.lnet = model
        self.lnet.eval()

    def __call__(self, **policy_kwargs):
        array = policy_kwargs['synthetic_array']
        self.lnet.eval()
        a = self.lnet.choose_action(v_wrap(array[None, None, :]))
        # print('\n',a)
        action = ACTION_MAP[a]


        return action
