# @title:    model.py
# @author:   Jan Frederik Liebig
# @date:     07.09.2021

############################################################
# Imports
import torch.nn as nn
from src.utility.container import ModelContainer
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
from typing import Any

############################################################
# Code


class PPOModel:
    """
    Container for the models used in the ppo algorithm
    """

    actor: Any
    actor_old: Any
    loss: Any


class PPONetwork(nn.Module):
    def __init__(self, model_container: ModelContainer, sample):
        """
        Initializes a new model for 2d stacked frames and single frame type
        @params:
            model_container utility container for the model
            sample  sample input data
        """
        super(PPONetwork, self).__init__()
        self.num_channel = model_container.num_channel
        self.num_input_x = model_container.num_inputs_x
        self.num_input_y = model_container.num_inputs_y
        self.transfer = False

        self.device = model_container.device
        self.conv1 = nn.Conv2d(
            in_channels=self.num_channel, out_channels=32, kernel_size=5
        ).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5).to(
            self.device
        )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5).to(
            self.device
        )

        self.num_actions = model_container.num_outputs
        self.name = model_container.name
        self._to_linear = None

        self.conv_layers(sample)

        self.d1 = nn.Dropout(0.01)
        self.fc1 = nn.Linear(self._to_linear, 128).to(self.device)
        self.act1 = nn.ReLU()
        self.d2 = nn.Dropout(0.01)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.act2 = nn.ReLU()
        self.d3 = nn.Dropout(0.01)
        self.fc3 = nn.Linear(64, 32).to(self.device)
        self.act3 = nn.ReLU()
        self.d4 = nn.Dropout(0.01)
        self.fc4 = nn.Linear(32, self.num_actions).to(self.device)
        self.act4 = nn.Softmax(dim=-1)

        self.actor = nn.Sequential(
            self.d1,
            self.fc1,
            self.act1,
            self.d2,
            self.fc2,
            self.act2,
            self.d3,
            self.fc3,
            self.act3,
            self.d4,
            self.fc4,
            self.act4,
        )

        self.c_d1 = nn.Dropout(0.01)
        self.c_fc1 = nn.Linear(self._to_linear, 128).to(self.device)
        self.c_act1 = nn.ReLU()
        self.c_d2 = nn.Dropout(0.01)
        self.c_fc2 = nn.Linear(128, 64).to(self.device)
        self.c_act2 = nn.ReLU()
        self.c_d3 = nn.Dropout(0.01)
        self.c_fc3 = nn.Linear(64, 32).to(self.device)
        self.c_act3 = nn.ReLU()
        self.c_d4 = nn.Dropout(0.01)
        self.c_fc4 = nn.Linear(32, 1).to(self.device)

        self.critic = nn.Sequential(
            self.c_d1,
            self.c_fc1,
            self.c_act1,
            self.c_d2,
            self.c_fc2,
            self.c_act2,
            self.c_d3,
            self.c_fc3,
            self.c_act3,
            self.c_d4,
            self.c_fc4,
        )

    def conv_layers(self, x):
        """
        Passes the input value through the convolutional layers
        @params
            x the input value
        @returns
            the result of the convolutional layers
        """
        x = x.view(-1, self.num_channel, self.num_input_x, self.num_input_y)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, state):
        """
        The forward pass through the model
        @params:
            state   the input state
        @return
            the output of the critic model
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        state_values = self.critic(x)
        return state_values

    def act(self, state):
        """
        Determines the next action following the input state
        @params
            state   the input state
        @returns
            the next action
            the logarithmic probability of the action
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """
        Evaluates a state with an action
        @params
            state   the input state
            action  the input action to evaluate
        @return
            the logarithmic probability of the action
            the state values
            the entropy of the distribution
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        action_probs = self.actor(x)
        if self.transfer:
            print(action_probs)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)

        return action_logprobs, state_values, dist_entropy


class PPONetwork3D(nn.Module):
    """
    Initializes a new model for 3d stacked frames
    @params:
        model_container utility container for the model
        sample  sample input data
    """

    def __init__(self, model_container: ModelContainer, sample):
        super(PPONetwork3D, self).__init__()
        self.num_channel = model_container.num_channel
        self.num_input_x = model_container.num_inputs_x
        self.num_input_y = model_container.num_inputs_y
        self.num_img = model_container.num_images

        self.device = model_container.device
        self.conv1 = nn.Conv3d(
            in_channels=self.num_channel, out_channels=32, kernel_size=(1, 5, 5)
        ).to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=(1, 5, 5)
        ).to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=(1, 5, 5)
        ).to(self.device)

        self.num_actions = model_container.num_outputs
        self.name = model_container.name
        self._to_linear = None

        self.conv_layers(sample)

        self.d1 = nn.Dropout(0.01)
        self.fc1 = nn.Linear(self._to_linear, 128).to(self.device)
        self.act1 = nn.ReLU()
        self.d2 = nn.Dropout(0.01)
        self.fc2 = nn.Linear(128, 64).to(self.device)
        self.act2 = nn.ReLU()
        self.d3 = nn.Dropout(0.01)
        self.fc3 = nn.Linear(64, 32).to(self.device)
        self.act3 = nn.ReLU()
        self.d4 = nn.Dropout(0.01)
        self.fc4 = nn.Linear(32, self.num_actions).to(self.device)
        self.act4 = nn.Softmax(dim=-1)

        self.actor = nn.Sequential(
            self.d1,
            self.fc1,
            self.act1,
            self.d2,
            self.fc2,
            self.act2,
            self.d3,
            self.fc3,
            self.act3,
            self.d4,
            self.fc4,
            self.act4,
        )

        self.c_d1 = nn.Dropout(0.01)
        self.c_fc1 = nn.Linear(self._to_linear, 128).to(self.device)
        self.c_act1 = nn.ReLU()
        self.c_d2 = nn.Dropout(0.01)
        self.c_fc2 = nn.Linear(128, 64).to(self.device)
        self.c_act2 = nn.ReLU()
        self.c_d3 = nn.Dropout(0.01)
        self.c_fc3 = nn.Linear(64, 32).to(self.device)
        self.c_act3 = nn.ReLU()
        self.c_d4 = nn.Dropout(0.01)
        self.c_fc4 = nn.Linear(32, 1).to(self.device)

        self.critic = nn.Sequential(
            self.c_d1,
            self.c_fc1,
            self.c_act1,
            self.c_d2,
            self.c_fc2,
            self.c_act2,
            self.c_d3,
            self.c_fc3,
            self.c_act3,
            self.c_d4,
            self.c_fc4,
        )

    def conv_layers(self, x):
        """
        Passes the input value through the convolutional layers
        @params
            x the input value
        @returns
            the result of the convolutional layers
        """
        x = x.view(
            -1, self.num_channel, self.num_img, self.num_input_x, self.num_input_y
        )
        x = F.max_pool3d(F.relu(self.conv1(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv3(x)), (1, 2, 2))
        if self._to_linear is None:
            self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, state):
        """
        The forward pass through the model
        @params:
            state   the input state
        @return
            the output of the critic model
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        state_values = self.critic(x)
        return state_values

    def act(self, state):
        """
        Determines the next action following the input state
        @params
            state   the input state
        @returns
            the next action
            the logarithmic probability of the action
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """
        Evaluates a state with an action
        @params
            state   the input state
            action  the input action to evaluate
        @return
            the logarithmic probability of the action
            the state values
            the entropy of the distribution
        """
        x = self.conv_layers(state)
        x = x.view(-1, self._to_linear)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)

        return action_logprobs, state_values, dist_entropy
