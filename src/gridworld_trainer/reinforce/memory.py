# @title:    memory.py
# @author:   Jan Frederik Liebig
# @date:     02.09.2021

############################################################
# Imports
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch

############################################################
# Code


class MemoryReinforce:
    def __init__(self, rollout_size, obs_size_x, obs_size_y, obs_channel, device):
        """
        Initializes the memory for the reinforce algorithm for 2d stacked frames and single frame type
        @params
            rollout_size    the maximum rollout size
            obs_size_x      the x size of the observation
            obs_size_y      the y size of the observation
            obs_channel     the number of channels in the observation
            device          the device used in the models
        """
        self.rollout_size = rollout_size
        self.obs_size_x = obs_size_x
        self.obs_size_y = obs_size_y
        self.obs_channel = obs_channel
        self.device = device
        self.reset()

    def insert(self, step, done, action, log_prob, reward, obs):
        """
        Inserts new data in the memory
        @params:
            step        the current step
            done        true if the state is terminal
            action      the used action
            log_prob    the logarithmic probability of the action
            reward      the received reward
            obs         the observation to insert
        """
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)
        self.obs[step] = self.obs[step].to(self.device)

    def reset(self):
        """
        Resets the memory
        """
        self.done = torch.zeros(self.rollout_size, 1)
        self.returns = torch.zeros(self.rollout_size + 1, 1, requires_grad=False)
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64)
        self.log_probs = torch.zeros(self.rollout_size, 1)
        self.rewards = torch.zeros(self.rollout_size, 1)
        self.obs = torch.zeros(
            self.rollout_size, self.obs_channel, self.obs_size_x, self.obs_size_y
        )
        self.obs = self.obs.to(self.device)

    def compute_returns(self, gamma):
        """
        Computes the returns for each episode
        @params:
            gamma   the discount factor
        """
        self.last_done = (self.done == 1).nonzero().max()
        self.returns[self.last_done + 1] = 0.0

        for step in reversed(range(self.last_done + 1)):
            self.returns[step] = (
                self.returns[step + 1] * gamma * (1 - self.done[step])
                + self.rewards[step]
            )

    def batch_sampler(self, batch_size):
        """
        Samples a batch with the data
        @params:
            batch_size  the size of the requested batch
        """
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)), batch_size, drop_last=True
        )
        for indices in sampler:
            yield self.actions[indices], self.returns[indices], self.obs[indices]


class MemoryReinforce3D:
    def __init__(
        self, rollout_size, obs_size_x, obs_size_y, obs_size_z, obs_channel, device
    ):
        """
        Initializes the memory for the reinforce algorithm for 3d stacked frames
        @params
            rollout_size    the maximum rollout size
            obs_size_x      the x size of the observation
            obs_size_y      the y size of the observation
            obs_size_z      the z size of the observation
            obs_channel     the number of channels in the observation
            device          the device used in the models
        """
        self.rollout_size = rollout_size
        self.obs_size_x = obs_size_x
        self.obs_size_y = obs_size_y
        self.obs_size_z = obs_size_z
        self.obs_channel = obs_channel
        self.device = device
        self.reset()

    def insert(self, step, done, action, log_prob, reward, obs):
        """
        Inserts new data in the memory
        @params:
            step        the current step
            done        true if the state is terminal
            action      the used action
            log_prob    the logarithmic probability of the action
            reward      the received reward
            obs         the observation to insert
        """
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)
        self.obs[step] = self.obs[step].to(self.device)

    def reset(self):
        """
        Resets the memory
        """
        self.done = torch.zeros(self.rollout_size, 1)
        self.returns = torch.zeros(self.rollout_size + 1, 1, requires_grad=False)
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64)
        self.log_probs = torch.zeros(self.rollout_size, 1)
        self.rewards = torch.zeros(self.rollout_size, 1)
        self.obs = torch.zeros(
            self.rollout_size,
            self.obs_channel,
            self.obs_size_z,
            self.obs_size_x,
            self.obs_size_y,
        )
        self.obs = self.obs.to(self.device)

    def compute_returns(self, gamma):
        """
        Computes the returns for each episode
        @params:
            gamma   the discount factor
        """
        self.last_done = (self.done == 1).nonzero().max()
        self.returns[self.last_done + 1] = 0.0

        for step in reversed(range(self.last_done + 1)):
            self.returns[step] = (
                self.returns[step + 1] * gamma * (1 - self.done[step])
                + self.rewards[step]
            )

    def batch_sampler(self, batch_size):
        """
        Samples a batch with the data
        @params:
            batch_size  the size of the requested batch
        """
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)), batch_size, drop_last=True
        )
        for indices in sampler:
            yield self.actions[indices], self.returns[indices], self.obs[indices]
