import torch
import gym
import numpy as np
from torch import nn
from model.Networks import DeepQNet, DuelDeepQNet, C51DeepQNet


import IPython.terminal.debugger as Debug


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class RainbowAgent(object):
    # initialize the agent
    def __init__(self,
                 env_params=None,
                 agent_params=None
                 ):
        # save the parameters
        self.env_params = env_params
        self.agent_params = agent_params

        # environment parameters
        self.action_space = np.linspace(0, env_params['act_num'], env_params['act_num'], endpoint=False).astype('uint8')
        self.action_dim = env_params['act_num']
        self.obs_dim = env_params['obs_dim']

        # create behavior policy and target networks
        self.dqn_mode = agent_params['dqn_mode']
        self.gamma = agent_params['gamma']

        # whether using the dueling network or not
        if agent_params['use_dueling']:
            self.behavior_policy_net = DuelDeepQNet(self.obs_dim, self.action_dim)
            self.target_policy_net = DuelDeepQNet(self.obs_dim, self.action_dim)
        elif agent_params['use_distributional']:
            self.val_max = agent_params['v_max']
            self.val_min = agent_params['v_min']
            self.atoms_num = agent_params['atoms_num']
            self.behavior_policy_net = C51DeepQNet(self.obs_dim, self.action_dim, self.atoms_num)
            self.target_policy_net = C51DeepQNet(self.obs_dim, self.action_dim, self.atoms_num)
        else:
            self.behavior_policy_net = DeepQNet(self.obs_dim, self.action_dim)
            self.target_policy_net = DeepQNet(self.obs_dim, self.action_dim)

        # initialize target network with behavior network
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(agent_params['device'])
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=self.agent_params['lr'])

        # other parameters
        self.eps = 1

    # get action
    def get_action(self, obs):
        if np.random.random() < self.eps:  # with probability eps, the agent selects a random action
            action = np.random.choice(self.action_space, 1)[0]
        else:  # with probability 1 - eps, the agent selects a greedy policy
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                if self.agent_params['use_distributional']:
                    q_value_dist = self.behavior_policy_net(obs)
                    expected_q_values = q_value_dist * torch.linspace(self.val_min, self.val_max, self.atoms_num)
                    action = expected_q_values.sum(dim=2).max(dim=1)[1].item()
                else:
                    q_values = self.behavior_policy_net(obs)
                    action = q_values.max(dim=1)[1].item()
        return int(action)

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        if self.agent_params['use_per']:
            batch_data_tensor, per_weights = self._batch_to_tensor(batch_data)
        else:
            batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        if self.agent_params['use_distributional']:
            batch_size = obs_tensor.shape[0]
            # compute the current distribution
            current_dist = self.behavior_policy_net(obs_tensor)
            actions_tensor = actions_tensor.unsqueeze(dim=1).expand(batch_size, 1, self.atoms_num)
            current_dist = current_dist.gather(dim=1, index=actions_tensor.long()).squeeze(dim=1)
            current_dist = current_dist.clamp_(1e-3, 0.999)
            # compute the projected target distribution
            with torch.no_grad():
                proj_dist = self._distribution_projection(next_obs_tensor, rewards_tensor, dones_tensor)
                proj_dist = proj_dist.clamp_(1e-3, 0.999)

            # compute the cross entropy loss
            loss = -1 * (proj_dist.detach() * current_dist.log()).sum(dim=1).mean()
        else:
            # compute the q value estimation using the behavior network
            pred_q_value = self.behavior_policy_net(obs_tensor)
            pred_q_value = pred_q_value.gather(dim=1, index=actions_tensor)

            # compute the TD target using the target network
            if self.dqn_mode == 'vanilla':
                # compute the TD target using vanilla method: TD = r + gamma * max a' Q(s', a')
                # no gradient should be tracked
                with torch.no_grad():
                    max_next_q_value = self.target_policy_net(next_obs_tensor).max(dim=1)[0].view(-1, 1)
                    td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
            else:
                # compute the TD target using double method: TD = r + gamma * Q(s', argmaxQ_b(s'))
                with torch.no_grad():
                    max_next_actions = self.behavior_policy_net(next_obs_tensor).max(dim=1)[1].view(-1, 1).long()
                    max_next_q_value = self.target_policy_net(next_obs_tensor).gather(dim=1, index=max_next_actions).view(
                        -1, 1)
                    td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
                    td_target_value = td_target_value.detach()

            # compute the loss
            if self.agent_params['use_per']:
                # NOTE: different the td error is weighted by importance
                loss = ((pred_q_value - td_target_value) ** 2) * per_weights
                prios = loss.detach().cpu().numpy() + 1e-5
                loss = torch.mean(loss)
            else:
                loss = torch.nn.functional.mse_loss(pred_q_value, td_target_value)

        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.agent_params['use_per']:
            return loss.item(), prios
        else:
            return loss.item()

    # update update target policy
    def update_target_policy(self):
        if self.agent_params['use_soft_update']:  # update the target network using polyak average (i.e., soft update)
            # polyak ~ 0.95
            for param, target_param in zip(self.behavior_policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.agent_params['polyak']) * param + self.agent_params['polyak'] * target_param)
        else:  # hard update
            self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        if self.agent_params['use_per']:
            obs_arr, action_arr, reward_arr, next_obs_arr, done_arr, _, weights = batch_data
        else:
            obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data

        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        if self.agent_params['use_per']:
            weights_tensor = torch.from_numpy(weights).float().view(-1, 1).to(self.device)
            return batch_data_tensor, weights_tensor
        else:
            return batch_data_tensor

    def _distribution_projection(self, next_states, rewards, dones):
        with torch.no_grad():
            # get the batch size for reshaping
            batch_size = next_states.shape[0]
            # delta z
            delta_z = float(self.val_max - self.val_min) / (self.atoms_num - 1)
            support = torch.linspace(self.val_min, self.val_max, self.atoms_num)

            # compute the distribution of the next state
            if self.agent_params['dqn_mode'] == "vanilla":
                next_dist = self.target_policy_net(next_states)
                expected_q_value = (next_dist * support).sum(dim=2)
            else:
                next_dist = self.behavior_policy_net(next_states)
                expected_q_value = (next_dist * support).sum(dim=2)

            # get the best next action
            best_next_action = expected_q_value.max(dim=1)[1]
            best_next_action = best_next_action.unsqueeze(dim=1).unsqueeze(dim=1).expand(batch_size, 1, self.atoms_num)

            # select the best nest distribution
            next_dist = self.target_policy_net(next_states)
            next_dist = next_dist.gather(dim=1, index=best_next_action).squeeze(1)

            # expand the others
            rewards = rewards.expand_as(next_dist)
            dones = dones.expand_as(next_dist)
            support = support.unsqueeze(dim=0).expand_as(next_dist)

            Tz = rewards + (1 - dones) * self.gamma * support
            Tz = Tz.clamp(min=self.val_min, max=self.val_max)
            b = (Tz - self.val_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(0, (batch_size - 1) * self.atoms_num, batch_size).long() \
                .unsqueeze(1).expand(batch_size, self.atoms_num)

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            return proj_dist