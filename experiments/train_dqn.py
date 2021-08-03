from utils.Schedule import LinearSchedule, ExponentialSchedule, BetaSchedule
from utils.MemoryBuffer import DQNReplayBuffer, NaivePrioritizedBuffer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm

import IPython.terminal.debugger as Debug


class DQNExperiment(object):
    def __init__(self, agent, env, test_env, trn_params):
        # initialize the experiment
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.trn_params = trn_params

        # training parameters
        # self.schedule = LinearSchedule(1, 0.01, trn_params['total_time_steps'] / 3)
        self.schedule = ExponentialSchedule()
        if trn_params['use_per']:
            self.memory = NaivePrioritizedBuffer(capacity=trn_params['memory_size'])
            self.beta_schedule = BetaSchedule(beta_frames=100)
        else:
            self.memory = DQNReplayBuffer(trn_params['memory_size'])
        self.start_train_step = trn_params['start_train_step']
        self.total_time_steps = trn_params['total_time_steps']
        self.update_policy_freq = trn_params['update_policy_freq']
        self.update_target_freq = trn_params['update_target_freq']
        self.memory_size = trn_params['memory_size']
        self.batch_size = trn_params['batch_size']

        # special modifications
        self.use_her = trn_params['use_her']  # whether use her
        self.use_per = trn_params['use_per']  # whether use prioritized experience replay

        # save results
        self.trn_returns = []
        self.eval_returns = []
        self.save_dir = trn_params['save_dir']
        self.model_name = trn_params['model_name']

        # create the summary writer
        self.tb = SummaryWriter(comment=f"_{self.agent.agent_params['dqn_mode']}_")

    def reset(self):
        return self.env.reset()

    def get_action(self, obs):
        return self.agent.get_action(obs)

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        return next_obs, reward, done, {}

    def run(self):
        # training variables
        episode_t = 0
        episode_idx = 0
        rewards = []
        loss = 0

        # reset the environment
        obs = self.env.reset()

        # start training
        pbar = tqdm.trange(self.total_time_steps)
        for t in pbar:
            # get one action
            self.agent.eps = self.schedule.get_value(t)
            action = self.get_action(obs)
            # step in the environment
            next_obs, reward, done, _ = self.step(action)

            # # print info
            # print(f"Ep={episode_idx}: "
            #       f"state={obs['observation']} "
            #       f"action={ACTION_NAME[action]} "
            #       f"reward={reward} "
            #       f"next_state={next_obs['observation']} "
            #       f"desired_goal={obs['desired_goal']} "
            #       f"achieved_goal={next_obs['achieved_goal']}")

            # add to the buffer
            self.memory.add(obs, action, reward, next_obs, done)
            rewards.append(reward)

            # check termination
            if done:
                # compute the return
                # G = 0
                # for r in reversed(rewards):
                #     G = r + self.agent.gamma * G
                G = np.sum(rewards)

                # store the return
                self.trn_returns.append(G)
                episode_idx = len(self.trn_returns)

                # print the information
                pbar.set_description(
                    f"Ep={episode_idx} | "
                    f"G={np.mean(self.trn_returns[-10:]) if self.trn_returns else 0:.2f}"
                )

                self.tb.add_scalar("return", np.mean(self.trn_returns[-10:]), episode_idx)
                self.tb.add_scalar("loss", loss, episode_idx)

                # reset the environment
                episode_t, rewards = 0, []
                obs = self.reset()
            else:
                # increment
                obs = next_obs
                episode_t += 1

            if t > self.start_train_step:
                # update the behavior model
                if not np.mod(t, self.update_policy_freq):
                    # NOTE: update the priority with the latest td error
                    if self.trn_params['use_per']:
                        per_beta = self.beta_schedule.get_value(t)
                        batch_data = self.memory.sample_batch(self.batch_size, per_beta)
                        loss, prios = self.agent.update_behavior_policy(batch_data)
                        self.memory.update_priorities(batch_data[5], prios)
                    else:
                        batch_data = self.memory.sample_batch(self.batch_size)
                        loss = self.agent.update_behavior_policy(batch_data)

                # update the target model
                if not np.mod(t, self.update_target_freq):
                    self.agent.update_target_policy()

        # save the results
        self.tb.close()
