from qlearning_tiancheng_wang.q_state import calculate_state
import numpy as np


# get initial status
def get_initial_status(game):
    done = game.process.termination
    state = calculate_state(
        self_position=game.agent.position,
        common_observation=game.common_observation,
        setting=game.setting,
        synthetic_array=game.synthetic_array,
        blinky_p=game.blinky.position,
        inky_p=game.inky.position
    )

    return state, done


# training step wrapper
def training_step(game):
    game.run_one_step_without_graph()
    action = tuple(game.agent.direction_proposal)
    state = game.agent.policy.state
    reward = game.process.current_reward
    done = game.process.termination
    return action, state, reward, done


# training function
def train(game, alpha=0.01, gamma=0.99):
    q_table = game.agent.policy.q_table

    # game.is_training = False
    # game.agent.policy.random_mode = True
    previous_state, done = get_initial_status(game)
    # print('ps', previous_state)
    # print(game.agent.policy.random_mode)
    while not done:

        assert(game.agent.policy.random_mode == True)
        action, state, reward, done = training_step(game)

        max_state_qval = \
            max(q_table[state, act] for act in [(0, 1), (1, 0), (-1, 0), (0, -1)])
        q_table[previous_state, action] = \
            q_table[previous_state, action] + alpha * (
                    reward + gamma * max_state_qval - q_table[previous_state, action]
            )
        # q_table[previous_state, action] = \
        #     q_table[previous_state, action] + alpha * (
        #             reward - q_table[previous_state, action]
        #     )
        previous_state = state
    return game.process.reward
