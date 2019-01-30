import abc
import torch

from .trainer_state import TrainerState


class AbstractTrainer:
    def __init__(self):
        self.state = TrainerState()

    @abc.abstractmethod()
    def run(self, *args, **kwargs):
        pass
