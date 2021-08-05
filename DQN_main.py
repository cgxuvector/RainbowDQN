from agent.RainbowAgent import RainbowAgent
from experiments.train_dqn import DQNExperiment
import torch
import random
import numpy as np
import argparse

import gym
import IPython.terminal.debugger as Debug

"""
    Comment: There is no bug in the code. The big difference is using Linear schedule versus using the
             exponential schedule. However, such exploration strategy assume that the policy should be
             better and better through the time line. But this assumption is not necessarily true. 
             My question is: Is exponential schedule always better than linear one? 
             
    Notes about DQN:
             - Epsilon schedule with exponential schedule
             - Large memory buffer
             - Large batch size
             - Use double DQN
"""


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_mode", type=str, default="vanilla")
    parser.add_argument("--use_dueling", action="store_true", default=False)
    parser.add_argument("--use_per", action="store_true", default=False)
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--use_distributional", action="store_true", default=False)

    return parser.parse_args()


Args = parse_input()


# make the environment
def make_env(params):
    # create the environment
    env = gym.make(params['env'])
    return env


# env params
env_params = {
    'env': "CartPole-v0",
    'act_num': None,
    'act_dim': None,
    'obs_dim': None,

    'rnd_seed': 3124
}


# agent params
agent_params = {
    'dqn_mode': Args.dqn_mode,
    'gamma': 0.9995,
    'device': "cpu",
    'lr': 1e-3,
    'use_soft_update': False,
    'polyak': 0.005,
    'use_per': Args.use_per,
    'use_dueling': Args.use_dueling,
    'use_distributional': Args.use_distributional,
    'atoms_num': 21,
    'v_max': 200,
    'v_min': 0
}


# training params
train_params = {
    'total_time_steps': 500_000,
    'start_train_step': 1_000,
    'memory_size': 80_000,
    'update_policy_freq': 4,
    'update_target_freq': 3_000,
    'batch_size': 64,
    'use_her': False,
    'use_per': Args.use_per,
    'model_name': 'test_dqn',
    'save_dir': './results',

    'run_name': ""
}


if __name__ == '__main__':
    # set the random seed
    np.random.seed(env_params["rnd_seed"])
    random.seed(env_params["rnd_seed"])
    torch.manual_seed(env_params["rnd_seed"])

    # create environment
    trn_env = make_env(env_params)
    test_env = make_env(env_params)

    # set the environment parameters
    env_params['act_num'] = trn_env.action_space.n
    env_params['act_dim'] = trn_env.action_space.n,
    env_params['obs_dim'] = trn_env.observation_space.shape[0]

    # create the DQN agent
    my_agent = RainbowAgent(env_params, agent_params)

    # set the experiment running name
    name = [agent_params['dqn_mode']]
    if agent_params['use_per']:
        name.append("PER")

    if agent_params['use_dueling']:
        name.append("DUELING")

    if agent_params['use_distributional']:
        name.append("C51")
    name = "_".join(name)
    train_params['run_name'] = name

    # create experiment
    my_experiment = DQNExperiment(my_agent, trn_env, test_env, train_params)

    # run the experiment
    my_experiment.run()