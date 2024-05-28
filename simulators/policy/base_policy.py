from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np


class BasePolicy(ABC):

    def __init__(self, id: str, config) -> None:
        super().__init__()
        self.id = id
        self.config = config

    @abstractmethod
    def get_action(
        self, obs: np.ndarray,
        agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """Gets the action to execute.

        Args:
            obs (np.ndarray): current observation.
            agents_action (Optional[Dict]): other agents' actions that are
                observable to the ego agent.

        Returns:
            np.ndarray: the action to be executed.
            dict: info for the solver, e.g., processing time, status, etc.
        """
        raise NotImplementedError

    def report(self):
        print(self.id)
        if self.policy_observable is not None:
            print("  - The policy can observe:", end=' ')
            for i, k in enumerate(self.policy_observable):
                print(k, end='')
                if i == len(self.policy_observable) - 1:
                    print('.')
                else:
                    print(', ', end='')
        else:
            print("  - The policy can only access observation.")
