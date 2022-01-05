# @title:    container.py
# @author:   Jan Frederik Liebig
# @date:     15.06.2021

# Imports
import torch.optim as optim
from dataclasses import dataclass
from typing import Any
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from enum import Enum


# Code


class FrameType(Enum):
    Single = 0
    Stacked2D = 1
    Stacked3D = 2


class Averager:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Resets the values
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the values
        @param:
            val the value to update with
            n   the number this value should be added
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class MinMax:
    def __init__(self, input, fixed: bool = True):
        """
        Initializes the MinMax class
            @param:
                input: Array of the following form: [value, minimum, maximum]
                fixed: flag if is is a fixed value, default True
        """
        self.value = input[0]
        self.minimum = input[1]
        self.maximum = input[2]
        self.fixed = fixed

    def __getitem__(self, item):
        if item == 0:
            return self.value
        if item == 1:
            return self.minimum
        if item == 2:
            return self.maximum
        return None

    def __add__(self, other):
        return MinMax([self.value + other, self.minimum, self.maximum])

    def __iadd__(self, other):
        return MinMax([self.value + other, self.minimum, self.maximum])

    def __mul__(self, other):
        return MinMax([self.value * other, self.minimum, self.maximum])

    def __sub__(self, other):
        return MinMax([self.value - other, self.minimum, self.maximum])

    def __mod__(self, other):
        return MinMax([self.value % other, self.minimum, self.maximum])

    def __truediv__(self, other):
        return MinMax([self.value / other, self.minimum, self.maximum])

    def __call__(self, *args, **kwargs):
        return self.value

    def __str__(self):
        return f"MinMax[value: {self.value}; minimum: {self.minimum}; maximum: {self.maximum}]"

    def check(self):
        """
        Checks if the value is in between minimum and maximum
            @return:
                True if the value is smaller than the maximum and larger than the minimum
                False, otherwise
        """
        if self.minimum > self.maximum:
            return False
        if self.minimum > self.value:
            return False
        if self.value > self.maximum:
            return False
        return True

    def check_value(self, value):
        """
        Checks if a given value is in between minimum and maximum
            @return:
                True if the value is smaller than the maximum and larger than the minimum
                False, otherwise
        """
        if self.minimum > value:
            return False
        if value > self.maximum:
            return False
        return True

    def set(self, x):
        """
        Sets the value to a given value
            @param:
                x: the value to set
        """
        if self.fixed:
            assert True, "Cannot change fixed parameter"
        else:
            self.value = x

    def sample(self, seed=0):
        """
        Generates a random sample in the range min - max
            @return:
                A random value in the set range
        """
        np.random.seed(seed)
        if type(self.value) == int:
            return int(
                (self.maximum - self.minimum) * np.random.random_sample() + self.minimum
            )
        else:
            return (
                self.maximum - self.minimum
            ) * np.random.random_sample() + self.minimum

    def mutate(self, cut):
        """
        Mutates the value random in a range
            @param:
                cut: the percentage cut to mutate the value
        """
        if not self.check():
            self.value = self.sample()
        done = False
        while not done:
            random = np.random.uniform(-cut, cut) + 1
            value = self.value * random
            if self.check_value(value):
                if type(self.value) == int:
                    self.value = int(value)
                else:
                    self.value = value
                done = True


@dataclass
class UtilityContainer:
    """
    A container with utility information for the model
    """

    environment_id: str = ""
    writer: SummaryWriter = None
    log_iterations = MinMax([2000, 50, 10_000])
    seed: Any = None
    path: str = None
    loading: bool = False
    save: bool = False
    logging: bool = True
    init_logging: bool = True
    frame_type: FrameType = FrameType.Single


@dataclass
class ModelContainer:
    """
    A container with utility information for the model
    """

    name: str = ""
    num_inputs_x: int = 0
    num_inputs_y: int = 0
    num_channel: int = 3
    num_images: int = 1
    num_outputs: int = 0
    device: Any = None


@dataclass
class ModelSavingContainer:
    """
    A container for saving a model
    """

    model_container = None
    optimizer_state = None


@dataclass
class ReinforceContainer:
    def __init__(self):
        """
        Hyperparameter container for the reinforce algorithm
        """
        self.max_memory_size: MinMax = MinMax([2500, 200, 10_000])  # fixed
        self.num_updates: MinMax = MinMax([50, 1, 1_000])  # fixed
        self.batch_size: MinMax = MinMax([1024, 8, 2_048])  # fixed

        self.optimizer = optim.Adam
        self.optimizer_name = self.optimizer.__name__

        self.policy_epochs: MinMax = MinMax([10, 3, 100], False)
        self.entropy_coefficient: MinMax = MinMax([0.001, 0.0, 0.99], False)

        self.learning_rate: MinMax = MinMax([0.001, 0.0, 2.0], False)
        self.discount: MinMax = MinMax([0.99, 0.0, 0.999], False)
        self.weight_decay: MinMax = MinMax([0.00001, 0.0, 0.001], False)
        self.momentum_sgd: MinMax = MinMax([0.9, 0.85, 0.99], False)


@dataclass
class DQNContainer:
    def __init__(self):
        """
        Hyperparameter container for the dqn algorithm
        """
        self.batch_size: MinMax = MinMax([1024, 8, 2_048])  # fixed
        self.environment_steps: MinMax = MinMax([100_000, 10_000, 500_000])  # fixed
        self.epsilon: MinMax = MinMax([0.99, 0.0001, 0.9999])  # fixed
        self.epsilon_start_decay: MinMax = MinMax(
            [1, 0, self.environment_steps[2]]
        )  # fixed
        self.epsilon_end_decay: MinMax = MinMax(
            [self.environment_steps() / 2, 1, self.environment_steps()]
        )  # fixed
        self.epsilon_decay: MinMax = MinMax(
            [
                self.epsilon()
                / (self.epsilon_end_decay() - self.epsilon_start_decay()),
                0.0,
                1.0,
            ]
        )  # fixed
        self.target_update: MinMax = MinMax([100, 1, 10_000])  # fixed
        self.training_steps: MinMax = MinMax([32, 1, 10_000])  # fixed
        self.replay_memory_size: MinMax = MinMax([10_000, 200, 100_000])  # fixed
        self.min_replay_memory_size: MinMax = MinMax([2000, 200, 10_000])  # fixed

        self.optimizer = optim.Adam
        self.optimizer_name = self.optimizer.__name__

        self.learning_rate: MinMax = MinMax([0.001, 0.0, 0.2], False)
        self.discount: MinMax = MinMax([0.95, 0.0, 0.999], False)
        self.weight_decay: MinMax = MinMax([0.00001, 0.0, 0.001], False)
        self.momentum_sgd: MinMax = MinMax([0.9, 0.85, 0.99], False)


@dataclass
class ProximalPolicyOptimizationContainer:
    def __init__(self):
        """
        Hyperparameter container for the ppo algorithm
        """
        self.num_epochs: MinMax = MinMax([50, 10, 10_000])
        self.environment_steps: MinMax = MinMax([100_000, 500, 1_000_000])
        self.episode_update: MinMax = MinMax([2_000, 1, 10_000])
        self.eps_clip: MinMax = MinMax([0.2, 0.0, 1.0])

        self.optimizer = optim.Adam
        self.optimizer_name = self.optimizer.__name__

        self.learning_rate_actor: MinMax = MinMax([0.001, 0.0, 1.0], False)
        self.learning_rate_critic: MinMax = MinMax([0.001, 0.0, 1.0], False)
        self.discount: MinMax = MinMax([0.95, 0.0, 0.999], False)
        self.weight_decay: MinMax = MinMax([0.00001, 0.0, 0.001], False)
        self.momentum_sgd: MinMax = MinMax([0.9, 0.85, 0.99], False)


@dataclass
class ActorCriticContainer:
    def __init__(self):
        """
        Hyperparameter container for the actor critic algorithm
        """
        self.num_epochs: MinMax = MinMax([50, 10, 10_000])
        self.environment_steps: MinMax = MinMax([100_000, 500, 1_000_000])
        self.episode_update: MinMax = MinMax([2_000, 1, 10_000])
        self.eps_clip: MinMax = MinMax([0.2, 0.0, 1.0])

        self.optimizer = optim.Adam
        self.optimizer_name = self.optimizer.__name__

        self.learning_rate_actor: MinMax = MinMax([0.001, 0.0, 1.0], False)
        self.learning_rate_critic: MinMax = MinMax([0.001, 0.0, 1.0], False)
        self.discount: MinMax = MinMax([0.95, 0.0, 0.999], False)
        self.weight_decay: MinMax = MinMax([0.00001, 0.0, 0.001], False)
        self.momentum_sgd: MinMax = MinMax([0.9, 0.85, 0.99], False)


@dataclass
class StatisticContainer:
    def __init__(self):
        """
        Dataclass container for the statistics
        Including the following parameters
            success:    counter for the successful runs
            count:  counter for the total runs
            avg_reward: contains the average reward
            avg_steps: contains the average steps
            avg_reward_penalty: contains the average reward penalty
        """
        self.success: int = 0
        self.count: int = 0
        self.avg_reward: Averager = Averager()
        self.avg_steps: Averager = Averager()
        self.avg_reward_penalty: Averager = Averager()


@dataclass
class PopulationContainer:
    """
    Dataclass container for the population based training
    Including the following parameters
        name:   Name of the population
        seed:   Initial seed for the random generators

        population_size:    size of the population
        evaluation_steps:   parameter for the next evaluation

        cut:    percentage of the population that gets exploited
        mutation_cut:   max percentage of mutation in a single step
        history_score_cut:  factor for the previous score
    """

    def __init__(self):
        self.name: str = ""
        self.seed: Any = None

        self.population_size: int = 10
        self.evaluation_steps: int = 20

        self.cut: float = 0.2
        self.mutation_cut = 0.2
        self.history_score_cut = 0.05


@dataclass
class PopulationModelContainer:
    """
    Dataclass container for a single agent in the population based training
    Including the following parameters:
        id: ID of the model
        model:  the model
        evaluation: container for evaluation statistics
        statistics: container for long therm logging statistics
        iteration_statistics:   container for short therm logging statistics
        optimizer:  the optimizer for this model
        score:  last calculated score
        hyper_container:    container containing the hyperparameter
        score_history:  the current score history
        path:   the logging path for this model
        writer: the board writer for this model
        parent: the last parent from exploitation
    """

    def __init__(self):
        self.id: int = 0
        self.model: Any = None
        self.evaluation: StatisticContainer = StatisticContainer()
        self.statistics: StatisticContainer = StatisticContainer()
        self.iteration_statistics: StatisticContainer = StatisticContainer()
        self.optimizer = None
        self.score: [float] = [0.0]
        self.hyper_container: Any = []
        self.score_history: float = 0.00
        self.path: str = "/" + str(self.id)
        self.writer = None
        self.parent: [int] = [-1]


@dataclass
class ImageContainer:
    """
    Container for three images, used in stacked frame methods
    """

    img_1 = None
    img_2 = None
    img_3 = None
