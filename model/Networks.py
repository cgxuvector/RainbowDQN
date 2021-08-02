from torch import nn
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



