from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import random
import numpy as np
import gym

from .utils import GenericAction, GenericState


class BaseEnv(gym.Env, ABC):

    def __init__(self, config_env) -> None:
        gym.Env.__init__(self)
        self.cnt = 0
        self.timeout = config_env.TIMEOUT
        self.end_criterion = config_env.END_CRITERION

    @abstractmethod
    def step(self,
             action: GenericAction) -> Tuple[GenericState, float, bool, Dict]:
        raise NotImplementedError

    @abstractmethod
    def get_obs(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(
        self, state: Optional[GenericState] = None, cast_torch: bool = False,
        **kwargs
    ) -> GenericState:
        """
        Resets the environment and returns the new state.

        Args:
            state (Optional[GenericState], optional): reset to this state if
                provided. Defaults to None.
            cast_torch (bool): cast state to torch if True.

        Returns:
            np.ndarray: the new state.
        """
        self.cnt = 0

    @abstractmethod
    def render(self):
        raise NotImplementedError

    def seed(self, seed: int = 0) -> None:
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
        random.seed(self.seed_val)
        """
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    @abstractmethod
    def report(self):
        raise NotImplementedError
