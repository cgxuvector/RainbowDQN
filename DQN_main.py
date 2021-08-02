from agent.DQNAgent import DQNAgent
from experiments.train_dqn import DQNExperiment
import torch
import random
import numpy as np

import gym


# make the environment
def make_env():
    # create the environment
    env = gym.make("CartPole-v0")
    return env

# env params

# agent params


if __name__ == '__main__':
    # init the parameters
    env_params, agent_params, train_params = parse_dqn_input()

    # set the random seed
    np.random.seed(env_params["rnd_seed"])
    random.seed(env_params["rnd_seed"])
    torch.manual_seed(env_params["rnd_seed"])

    # create environment
    trn_env = make_env(env_params)
    test_env = make_env(env_params)

    # create the agent
    my_agent = DQNAgent(env_params, agent_params)

    # create experiment
    my_experiment = DQNExperiment(my_agent, trn_env, test_env, train_params)

    # run the experiment
    my_experiment.run()