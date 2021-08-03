from agent.DQNAgent import DQNAgent
from agent.PERDQNAgent import PERDQNAgent
from agent.C51DQNAgent import C51DQNAgent
from experiments.train_dqn import DQNExperiment
import torch
import random
import numpy as np

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

# implement the Multi-steps training


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
    'dqn_mode': "double",
    'gamma': 0.995,
    'device': "cpu",
    'lr': 1e-3,
    'use_soft_update': False,
    'polyak': 0.005,
    'use_dueling': False,
    'atoms_num': 51
}


# training params
train_params = {
    'total_time_steps': 500_000,
    'start_train_step': 1_000,
    'memory_size': 100_000,
    'update_policy_freq': 4,
    'update_target_freq': 2_000,
    'batch_size': 256,
    'use_her': False,
    'use_per': False,
    'use_distributional': True,

    'model_name': 'test_dqn',
    'save_dir': './results'
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

    # create the agent
    if train_params['use_per']:
        my_agent = PERDQNAgent(env_params, agent_params)
    elif train_params['use_distributional']:
        my_agent = C51DQNAgent(env_params, agent_params)
    else:
        my_agent = DQNAgent(env_params, agent_params)

    # create experiment
    my_experiment = DQNExperiment(my_agent, trn_env, test_env, train_params)

    # run the experiment
    my_experiment.run()