import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch

class HorizonTaskWilson(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, num_actions, reward_scaling, reward_std, num_forced_choice, batch_size=1):
        self.num_actions = num_actions

        self.num_forced_choice = num_forced_choice
        self.reward_scaling = reward_scaling
        self.batch_size = batch_size
        self.device = None
        self.reward_std = reward_std

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.ones(self.num_actions + 1), np.ones(self.num_actions + 1))

    def reset(self, unequal=None, horizon=None):
        self.t = 0

        if horizon is None:
            self.max_steps =  np.random.choice([1, 6])
        else:
            self.max_steps = horizon

        self.mean_reward = torch.empty(self.batch_size, self.num_actions, device=self.device)
        for i in range(self.batch_size):
            idx = np.random.choice([0, 1])
            rew = np.random.choice([40, 60])
            div = np.random.choice([-30, -20, -12, -8, -4, 4, 8, 12, 20, 30])
            self.mean_reward[i, idx] = rew
            self.mean_reward[i, 1 - idx] = rew + div

        self.rewards = torch.round(Normal(self.mean_reward, self.reward_std).sample((self.num_forced_choice + self.max_steps,)))

        return self.forced_choice_data(unequal)

    def forced_choice_data(self, unequal):
        # randomly select actions
        self.action = torch.zeros(self.num_forced_choice, self.batch_size, device=self.device).long()
        if unequal is None:
            unequal = np.random.choice([True, False], size=self.batch_size)
        for i in range(self.batch_size):
            if unequal[i]:
                # case unequal information
                options = np.array([
                    [0, 0, 0, 1],
                    [0, 1, 1, 1]])
                forced_choices = options[np.random.randint(options.shape[0])]
            else:
                # case equal information
                forced_choices = np.array([0, 0, 1, 1])

            np.random.shuffle(forced_choices)
            self.action[:, i] = torch.from_numpy(forced_choices).to(self.device)

        reward = torch.stack([self.rewards[t].gather(1, self.action[t].unsqueeze(1)).squeeze(1) for t in range(self.num_forced_choice)])
        reward = reward / self.reward_scaling
        time_step = self.max_steps * torch.ones(self.num_forced_choice, self.batch_size, 1).to(self.device)

        observation = torch.cat((
            reward.unsqueeze(-1),
            self.action.float().unsqueeze(-1),
            time_step
        ), dim=-1)

        return observation

    def step(self, action):
        # action is long
        self.t += 1
        done = True if (self.t >= self.max_steps) else False

        regrets = self.mean_reward.max(dim=1)[0] - self.mean_reward.gather(1, action.unsqueeze(1)).squeeze(1)
        reward = self.rewards[self.num_forced_choice + self.t - 1].gather(1, action.unsqueeze(1)).squeeze(1)
        reward = reward / self.reward_scaling
        time_step = self.max_steps * torch.ones(self.batch_size, 1).to(self.device)

        observation = torch.cat((
            reward.unsqueeze(-1),
            action.float().unsqueeze(-1),
            time_step
        ), dim=-1)

        return observation, reward, done, {'regrets': regrets.mean()}
