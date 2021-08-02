from torch import nn
import IPython.terminal.debugger as Debug


# class of deep neural network model
class DeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, obs):
        x = self.fc_layer(obs)
        return x



