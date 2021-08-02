from utils.Schedule import LinearSchedule
from utils.ExperienceReplay import DQNReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm

ACTION_NAME = ['left', 'down', 'right', 'up']


class DQNExperiment(object):
    def __init__(self, agent, env, test_env, trn_params):
        # initialize the experiment
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.trn_params = trn_params

        # training parameters
        self.use_obs = trn_params['use_obs']
        self.schedule = LinearSchedule(1, 0.01, trn_params['total_time_steps'] / 3)
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
        self.tb = SummaryWriter(comment=f"_step={self.env.max_step}_")

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

        # reset the environment
        obs = self.env.reset()

        # start training
        pbar = tqdm.trange(self.total_time_steps)
        start_pos = self.env.agent_pos
        goal_pos = self.env.goal_loc
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
            self.memory.add(obs['observation'], action, reward, next_obs['observation'], done)
            rewards.append(reward)

            # check termination
            if done or episode_t == self.env.max_step:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.agent.gamma * G

                # store the return
                self.trn_returns.append(G)
                episode_idx = len(self.trn_returns)

                # print the information
                pbar.set_description(
                    f"Ep={episode_idx} | "
                    f"G={np.mean(self.trn_returns[-10:]) if self.trn_returns else 0:.2f} | "
                    f"Eval={np.mean(self.eval_returns[-10:]) if self.eval_returns else 0:.2f} | "
                    f"Init={start_pos} | "
                    f"Goal={goal_pos}"
                )

                self.tb.add_scalar("return", np.mean(self.trn_returns[-10:]), episode_idx)

                # reset the environment
                episode_t, rewards = 0, []
                obs = self.reset()
                start_pos = self.env.agent_pos
                goal_pos = self.env.goal_loc
            else:
                # increment
                obs = next_obs
                episode_t += 1

            if t > self.start_train_step:
                # update the behavior model
                if not np.mod(t, self.update_policy_freq):
                    batch_data = self.memory.sample_batch(self.batch_size)
                    self.agent.update_behavior_policy(batch_data)

                # update the target model
                if not np.mod(t, self.update_target_freq):
                    self.agent.update_target_policy()

        # save the results
        self.save()
        self.tb.close()
