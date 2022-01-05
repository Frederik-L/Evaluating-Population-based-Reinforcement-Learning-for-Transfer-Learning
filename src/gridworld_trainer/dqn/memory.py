# @title:    memory.py
# @author:   Jan Frederik Liebig
# @date:     14.09.2021

############################################################
# Imports
from dataclasses import dataclass
from typing import Any
from collections import deque
import random

############################################################
# Code


@dataclass
class TransitionDQN:
    """
    The dataclass for saving a state transition
    """
    state: Any = None
    action: int = -1
    reward: float = -1.0
    next_state: Any = None
    done: bool = False
    info: Any = None


class MemoryDQN:
    def __init__(self, buffer_size=100000):
        """
        Initializes the memory for the dqn
        @params:
            buffer_size the max buffer size for the memory
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def insert(self, replay: TransitionDQN):
        """
        Inserts a new transition in the memory
        """
        self.buffer.append(replay)

    def sample(self, num_samples):
        """
        Generates a random sample of the transitions
        @params:
            num_samples the requested number of stransitions
        @returns:
            a sample of the transitions in the memory
        """
        assert num_samples <= len(self.buffer)
        return random.sample(self.buffer, num_samples)

    def __len__(self):
        return len(self.buffer)
