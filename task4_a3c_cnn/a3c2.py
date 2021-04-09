"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from a3ctwo.utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from a3ctwo.shared_adam import SharedAdam
from environment.main import PacMan
import os
from a3ctwo.policy import ACTION_MAP
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 100
GAMMA = 0.9
MAX_EP = 500000

# env = PacMan(maze_row_num=2, maze_column_num=2, maze_row_height=2, maze_column_width=2)
N_S = 100
N_A = 4

ROW_NUMBER = 2
HEIGHT = ROW_NUMBER*3 + 1


# get initial status
def get_initial_status(game):
    done = game.process.termination
    previous_state = game.synthetic_array
    return previous_state, done


# training step wrapper
def training_step(game, act):
    game.run_one_step_without_graph(instruction=act)
    state = game.synthetic_array
    reward = game.process.current_reward
    done = game.process.termination
    return state, reward, done


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.cnn = nn.Conv2d(1, 4, (3, 3))
        self.drop = nn.Dropout(p=0.2)
        # self.cnn2 = nn.Conv2d(4, 8, (3, 3))
        self.linear = nn.Linear(s_dim, s_dim)
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.cnn, self.linear, self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = torch.relu(self.cnn(x))
        # x = torch.relu(self.cnn2(x))
        # print(x.shape)
        # x = self.drop(x)



        x = x.view(-1, N_S)

        x = torch.tanh(self.linear(x))
        x = self.drop(x)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        """

        :param gnet:
        :param opt:
        :param global_ep:
        :param global_ep_r:
        :param res_queue:
        :param name: int
        """
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)  # local network
        self.env = PacMan(maze_row_num=ROW_NUMBER, maze_column_num=ROW_NUMBER,
                          maze_row_height=2, maze_column_width=2)
        torch.manual_seed(name)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            self.env.random_reset()
            s = self.env.synthetic_array
            s = s.reshape((1, 1, HEIGHT, HEIGHT))
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                # if self.name == 'w00':
                # self.env.render()

                a = self.lnet.choose_action(v_wrap(s))
                s_, reward, done = training_step(self.env, ACTION_MAP[a])
                # if done: reward = -1
                ep_r += reward
                buffer_a.append(a)

                s_ = s_.reshape((1, 1, HEIGHT, HEIGHT))

                buffer_s.append(s)
                buffer_r.append(reward)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # print(buffer_r)
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

                        break
                s = s_
                total_step += 1

            # print(total_step)
            # if total_step % 500 == 0:
            #     # print('step:', total_step)
            #     self.lnet.eval()
            #     win = 0
            #     rewards_dict = []
            #     living_time = []
            #     for i in range(100):
            #         l = 0
            #         self.env.random_reset()
            #         s = self.env.synthetic_array
            #         s = s.reshape((1, 1, 7, 7))
            #         while True:
            #             a = self.lnet.choose_action(v_wrap(s))
            #             s_, reward, done = training_step(self.env, ACTION_MAP[a])
            #             s_ = s_.reshape((1, 1, 7, 7))
            #             s = s_
            #             l += 1
            #             if done:
            #                 if self.env.process.win:
            #                     win += 1
            #                 rewards_dict.append(self.env.process.reward)
            #                 living_time.append(l)
            #                 break
            #     print('reward:', np.mean(rewards_dict))
            #     print('win:',win)
            #     print('living_time', np.mean(living_time))
            #     self.lnet.train()

            # return
        self.res_queue.put(None)


if __name__ == "__main__":

    load_path = 'a3ctwo/1cnn_dropout_0.2_channel_4.pt'
    save_path = 'a3ctwo/1cnn_dropout_0.2_channel_4.pt'
    gnet = Net(N_S, N_A)  # global network
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    try:
        state_dict = torch.load(load_path)
        gnet.load_state_dict(state_dict['model_state_dict'])
        opt.load_state_dict(state_dict['optimizer_state_dict'])
    except FileNotFoundError:
        print('new model')

    gnet.share_memory()  # share the global parameters in multiprocessing

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    state_dict = {
        'model_state_dict': gnet.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }
    if not os.path.isfile(save_path) or input('Checkpoint exist, return 1 to overwrite.') == "1":
        torch.save(state_dict, save_path)
    else:
        print('Model not saved.')

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
