from torch import nn
import torch
import math
import torch.nn.functional as F
import math

import IPython.terminal.debugger as Debug


"""
    Comment: smaller model (i.e., hidden = 128) learns faster while bigger model (hidden = 256)
    seems more robust?
"""


# class of deep neural network model
class DeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, obs):
        x = self.fc_layer(obs)
        return x


""" For Dueling DQN
"""


# class of deep neural network model
class DuelDeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DuelDeepQNet, self).__init__()
        # define the feature neural network
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True)
        )

        # define the V value head
        self.state_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # define the A value head
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, act_dim)
        )

    # forward function
    def forward(self, obs):
        # compute the feature
        feature = self.feature_net(obs)
        # compute the state value
        state_value = self.state_layer(feature)
        # compute the advantage value
        advantage_value = self.advantage_layer(feature)

        return state_value + advantage_value - advantage_value.mean()


""" C51 Neural Network
"""


class C51DeepQNet(nn.Module):
    def __init__(self, obs_dim, act_num, atoms_num=51):
        super(C51DeepQNet, self).__init__()
        # parameters
        self.obs_dim = obs_dim
        self.act_num = act_num
        self.atoms_num = atoms_num
        # define the linear layer
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, atoms_num * act_num)
        )

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.view(-1, self.atoms_num)
        x = x.view(-1, self.act_num, self.atoms_num)
        x = F.softmax(x, dim=2)
        return x


# """
#     For Noisy Network
# """
#
#
# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.4):
#         super(NoisyLinear, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#         self.std_init = std_init
#
#         self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
#
#         self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
#         self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
#         self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
#
#         self.reset_parameters()
#         self.reset_noise()
#
#     def forward(self, x):
#         if self.training:
#             weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
#             bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
#         else:
#             weight = self.weight_mu
#             bias = self.bias_mu
#
#         return F.linear(x, weight, bias)
#
#     def reset_parameters(self):
#         mu_range = 1 / math.sqrt(self.weight_mu.size(1))
#
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
#
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
#
#     def reset_noise(self):
#         epsilon_in = self._scale_noise(self.in_features)
#         epsilon_out = self._scale_noise(self.out_features)
#
#         self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
#         self.bias_epsilon.copy_(self._scale_noise(self.out_features))
#
#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         x = x.sign().mul(x.abs().sqrt())
#         return x



