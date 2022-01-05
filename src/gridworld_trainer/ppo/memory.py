# @title:    memory.py
# @author:   Jan Frederik Liebig
# @date:     16.09.2021

############################################################
# Imports

############################################################
# Code


class MemoryPPO:
    def __init__(self):
        """
        Initializes the memory for the ppo
        """
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """
        Clears the current memory
        """
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
