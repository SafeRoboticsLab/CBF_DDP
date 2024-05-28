from typing import Optional, Tuple, Dict, List
import copy
import numpy as np

# Dynamics.
from .dynamics.bicycle5d import Bicycle5D

from .costs.base_margin import BaseMargin

# Footprint.
from .footprint.circle import CircleFootprint

# Policy.
from .policy.base_policy import BasePolicy
from .policy.ilqr_policy import iLQR
from .policy.ilqr_filter_policy import iLQRSafetyFilter
from .policy.ilqr_reachavoid_policy import iLQRReachAvoid


class Agent:
    """A basic unit in our environments.

    Attributes:
        dyn (object): agent's dynamics.
        footprint (object): agent's shape.
        policy (object): agent's policy.
    """
    policy: Optional[BasePolicy]
    safety_policy: Optional[BasePolicy]
    ego_observable: Optional[List]
    agents_policy: Dict[str, BasePolicy]
    agents_order: Optional[List]

    def __init__(self, config, action_space: np.ndarray, env=None) -> None:
        if config.DYN == "Bicycle5D":
            self.dyn = Bicycle5D(config, action_space)
        else:
            raise ValueError("Dynamics type not supported!")

        try:
            self.env = copy.deepcopy(env)  # imaginary environment
        except Exception as e:
            print("WARNING: Cannot copy env - {}".format(e))

        if config.FOOTPRINT == "Circle":
            self.footprint = CircleFootprint(ego_radius=config.EGO_RADIUS)

        # Policy should be initialized by `init_policy()`.
        self.policy = None
        self.safety_policy = None
        self.id: str = config.AGENT_ID
        self.ego_observable = None
        self.agents_policy = {}
        self.agents_order = None

    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the next state of the vehicle given the current state and
        control input.

        Args:
            state (np.ndarray): (dyn.dim_x, ) array.
            control (np.ndarray): (dyn.dim_u, ) array.

        Returns:
            np.ndarray: next state.
            np.ndarray: clipped control.
        """
        assert control is not None, (
            "You need to pass in a control!"
        )

        return self.dyn.integrate_forward(
            state=state, control=control
        )

    def get_dyn_jacobian(
        self, nominal_states: np.ndarray, nominal_controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the linearized 'A' and 'B' matrix of the ego vehicle around
        nominal states and controls.

        Args:
            nominal_states (np.ndarray): states along the nominal trajectory.
            nominal_controls (np.ndarray): controls along the trajectory.

        Returns:
            np.ndarray: the Jacobian of next state w.r.t. the current state.
            np.ndarray: the Jacobian of next state w.r.t. the current control.
        """
        A, B = self.dyn.get_jacobian(nominal_states, nominal_controls)
        return np.asarray(A), np.asarray(B)

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
        if self.ego_observable is not None:
            for agent_id in self.ego_observable:
                assert agent_id in agents_action

        if agents_action is not None:
            _action_dict = copy.deepcopy(agents_action)
        else:
            _action_dict = {}

        _action, _solver_info = self.policy.get_action(  # Proposed action.
            obs=obs, agents_action=agents_action, **kwargs
        )
        _action_dict[self.id] = _action

        return _action, _solver_info

    def init_policy(
        self, policy_type: str, config, cost: Optional[BaseMargin] = None, **kwargs
    ):
        if policy_type == "iLQR":
            self.policy = iLQR(self.id, config, self.dyn, cost, **kwargs)
        elif policy_type == "iLQRReachAvoid":
            self.policy = iLQRReachAvoid(
                self.id, config, self.dyn, cost, **kwargs
            )
        elif policy_type == "iLQRSafetyFilter":
            self.policy = iLQRSafetyFilter(
                self.id, config, self.dyn, cost, **kwargs
            )
        else:
            raise ValueError(
                "The policy type ({}) is not supported!".format(policy_type)
            )

    def report(self):
        print(self.id)
        if self.ego_observable is not None:
            print("  - The agent can observe:", end=' ')
            for i, k in enumerate(self.ego_observable):
                print(k, end='')
                if i == len(self.ego_observable) - 1:
                    print('.')
                else:
                    print(', ', end='')
        else:
            print("  - The agent can only access observation.")

        if self.agents_order is not None:
            print("  - The agent keeps agents order:", end=' ')
            for i, k in enumerate(self.agents_order):
                print(k, end='')
                if i == len(self.agents_order) - 1:
                    print('.')
                else:
                    print(' -> ', end='')
