from abc import abstractmethod
from typing import Any, Tuple, Optional, Callable, List, Dict, Union
import numpy as np
from gym import spaces
from tqdm import tqdm

from .agent import Agent
from .base_env import BaseEnv

from .policy.base_policy import BasePolicy

class BaseSingleEnv(BaseEnv):
  """Implements an environment of a single agent.
  """

  def __init__(self, config_env: Any, config_agent: Any) -> None:
    super().__init__(config_env)
    self.env_type = "single-agent"

    # Action Space.
    action_space = np.array(config_agent.ACTION_RANGE, dtype=np.float32)
    self.action_dim = action_space.shape[0]
    self.action_dim_ctrl = action_space.shape[0]
    self.agent = Agent(config_agent, action_space)
    self.action_space = spaces.Box(
        low=action_space[:, 0], high=action_space[:, 1]
    )
    self.state_dim = self.agent.dyn.dim_x

    self.integrate_kwargs = getattr(config_env, "INTEGRATE_KWARGS", {})
    if "noise" in self.integrate_kwargs:
      if self.integrate_kwargs['noise'] is not None:
        self.integrate_kwargs['noise'] = np.array(
            self.integrate_kwargs['noise']
        )

  def step(
      self, action: np.ndarray, cast_torch: bool = False
  ) -> Tuple[np.ndarray, float, bool, Dict]:
    """Implements the step function in the environment.

    Args:
        action (np.ndarray).
        cast_torch (bool): cast state to torch if True.

    Returns:
        np.ndarray: next state.
        float: the reward that ctrl wants to maximize and dstb wants to
            minimize.
        bool: True if the episode ends.
        Dict[str, Any]]: additional information of the step, such as target
            margin and safety margin used in reachability analysis.
    """

    self.cnt += 1
    state_nxt = self.agent.integrate_forward(
        state=self.state, control=action, **self.integrate_kwargs
    )[0]
    state_cur = self.state.copy()
    self.state = state_nxt.copy()
    constraints = self.get_constraints(state_cur, action, state_nxt)
    cost = self.get_cost(state_cur, action, state_nxt, constraints)
    #targets = self.get_target_margin(state_cur, action, state_nxt)
    done, info = self.get_done_and_info(state_nxt, constraints, None)

    obs = self.get_obs(state_nxt)

    return obs, cost, done, info

  @abstractmethod
  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[Dict] = None
  ) -> float:
    """
    Gets the cost given current state, current action, and next state.

    Args:
        state (np.ndarray): current state.
        action (np.ndarray): current action.
        state_nxt (np.ndarray): next state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.

    Returns:
        float: the cost to minimize.
    """
    raise NotImplementedError

  @abstractmethod
  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all constaint functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states.
        action (np.ndarray): current actions.
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a constraint
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    """
    Gets the values of all target margin functions given current state, current
    action, and next state.

    Args:
        state (np.ndarray): current states.
        action (np.ndarray): current actions.
        state_nxt (np.ndarray): next state.

    Returns:
        Dict: each (key, value) pair is the name and value of a target margin
            function.
    """
    raise NotImplementedError

  @abstractmethod
  def get_done_and_info(
      self, state: np.ndarray, constraints: Dict, targets: Dict,
      final_only: bool = True, end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        state (np.ndarray): current state.
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    raise NotImplementedError

  def simulate_one_trajectory(
      self, T_rollout: int, end_criterion: str,
      reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      advanced_animate: bool=True, **kwargs
  ) -> Tuple[np.ndarray, int, Dict]:
    """
    Rolls out the trajectory given the horizon, termination criterion, reset
    keyword arguments, callback afeter every step, and callout after the
    rollout.

    Args:
        T_rollout (int): rollout horizon.
        end_criterion (str): termination criterion.
        reset_kwargs (Dict): keyword argument dictionary for reset function.
        action_kwargs (Dict): keyword argument dictionary for get_action
            function.
        rollout_step_callback (Callable): function to call after every step.
        rollout_episode_callback (Callable): function to call after rollout.

    Returns:
        np.ndarray: state trajectory.
        int: result (0: unfinished, 1: success, -1: failure).
        Dict: auxiliarry information -
            "action_history": action sequence.
            "plan_history": planning info for every step.
            "reward_history": rewards for every step.
            "step_history": information for every step.
    """
    # Stores the environment attributes and sets to rollout settings.
    timeout_backup = self.timeout
    end_criterion_backup = self.end_criterion
    self.timeout = T_rollout
    self.end_criterion = end_criterion

    state_history = []
    obs_history = []
    action_history = []
    reward_history = []
    plan_history = []
    step_history = []
    value_history = []
    process_time_history = []
    solver_iters_history = []
    complete_filter_indices = []
    barrier_filter_indices = []
    deviation_history = []
    # Initializes robot.
    controls_initialize = None

    result = 0
    obs = self.reset(**reset_kwargs)
    state_history.append(self.state)

    prev_sol=None
    for t in range(T_rollout):
      #kwargs['state'] = self.state.copy()
      action, solver_info = self.agent.get_action(
              obs=obs, controls=controls_initialize, prev_sol=prev_sol, state=self.state
          )
      prev_sol=solver_info

      if solver_info['mark_barrier_filter']:
        barrier_filter_indices.append(t)
      if solver_info['mark_complete_filter']:
        complete_filter_indices.append(t)
        
      # Applies action: `done` and `info` are evaluated at the next state.
      obs, reward, done, step_info = self.step(action)
      
      # Executes step callback and stores historyory.
      state_history.append(self.state)
      obs_history.append(obs)
      action_history.append(action)
      plan_history.append(solver_info)
      value_history.append(solver_info['marginopt'])
      reward_history.append(reward)
      step_history.append(step_info)
      process_time_history.append(solver_info['process_time'])
      solver_iters_history.append(solver_info['num_iters'])
      deviation_history.append(solver_info['deviation'])

      if advanced_animate:
        # We plot the safety plan where we enter the target set, then decelerate and stop to remain safe for infinite time for an infeasible task plan
        reachavoid_plan = solver_info['states']
        reachavoid_plan_ctrl = solver_info['controls']

        target_margins = self.cost.get_mapped_target_margin(reachavoid_plan, reachavoid_plan_ctrl)

        target_margins = np.array(target_margins)

        is_inside_target_index = np.argwhere(target_margins>=0).ravel()

        if is_inside_target_index.size==0:
          is_inside_target_index = 0
        else:
          is_inside_target_index = is_inside_target_index[0]

        target_plan = np.array( reachavoid_plan[:, 0:is_inside_target_index+1] )

        stopping_plan = self.simulate_stopping_plan( initial_state = np.array(reachavoid_plan[:, is_inside_target_index]), 
                                      stopping_ctrl=np.array([self.agent.dyn.ctrl_space[0, 0], 0]))

        safety_plan = np.concatenate((target_plan, np.array(stopping_plan).T), axis=1)
      else:
        safety_plan = None

      if rollout_step_callback is not None:
        rollout_step_callback(
            self, state_history, action_history, plan_history, step_history, safety_plan=safety_plan
        )

      # Checks termination criterion.
      if done:
        if step_info["done_type"] == "success":
          result = 1
        elif step_info["done_type"] == "failure":
          result = -1
        break

      controls_initialize = np.array(solver_info['reinit_controls'])

    if rollout_episode_callback is not None:
      rollout_episode_callback(
          self, state_history, action_history, plan_history, step_history, value_history=value_history, process_time_history=process_time_history, 
          solver_iters_history=solver_iters_history, deviation_history=deviation_history, 
          barrier_filter_indices=barrier_filter_indices, complete_filter_indices=complete_filter_indices, 
          label=self.agent.policy.filter_type
      )
    # Reverts to training setting.
    self.timeout = timeout_backup
    self.end_criterion = end_criterion_backup
    info = dict(
        obs_history=np.array(obs_history), action_history=np.array(action_history),
        plan_history=plan_history, reward_history=np.array(reward_history),
        step_history=step_history
    )

    return np.array(state_history), result, info
  
  def simulate_stopping_plan(self, initial_state: np.ndarray, stopping_ctrl: np.ndarray):
    """
      Simulates the stopping plan from an initial state by applying maximum deceleration.
      Deceleration policy renders the target set controlled invariant.
    """
    states = [initial_state]
    current_state = np.array(initial_state)
    while current_state[2]>self.agent.dyn.v_min:
      current_state, _ = self.agent.integrate_forward(current_state, stopping_ctrl)
      states.append(current_state)

    return states
  
  def simulate_task_plan(self, initial_state: np.ndarray, task_policy: BasePolicy, nsteps:int, is_ilqr: bool):
    """
      UNUSED: Simulates the task plan from an initial state
    """
    states = [initial_state]
    current_state = np.array(initial_state)
    idx = 0
    while idx<nsteps:
      idx = idx + 1
      if is_ilqr:
        task_ctrl = task_policy.get_action(current_state, None, state=current_state)
      else:
        task_ctrl = task_policy( current_state )
      current_state, _ = self.agent.integrate_forward(current_state, task_ctrl)
      states.append(current_state)

    return states

  def simulate_trajectories(
      self, num_trajectories: int, T_rollout: int, end_criterion: str,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None, return_info=False,
      **kwargs
  ):
    """
    Rolls out multiple trajectories given the horizon, termination criterion,
    reset keyword arguments, callback afeter every step, and callback after the
    rollout. Need to call env.reset() after this function to revert back to the
    training mode.
    """

    if isinstance(reset_kwargs_list, list):
      assert num_trajectories == len(reset_kwargs_list), (
          "The length of reset_kwargs_list does not match with",
          "the number of rollout trajectories"
      )
    if isinstance(action_kwargs_list, list):
      assert num_trajectories == len(action_kwargs_list), (
          "The length of action_kwargs_list does not match with",
          "the number of rollout trajectories"
      )

    results = np.empty(shape=(num_trajectories,), dtype=int)
    length = np.empty(shape=(num_trajectories,), dtype=int)
    trajectories = []
    info_list = []
    use_tqdm = kwargs.get('use_tqdm', False)
    if use_tqdm:
      iterable = tqdm(range(num_trajectories), desc='sim trajs', leave=False)
    else:
      iterable = range(num_trajectories)

    for trial in iterable:
      if isinstance(reset_kwargs_list, list):
        reset_kwargs = reset_kwargs_list[trial]
      else:
        reset_kwargs = reset_kwargs_list
      if isinstance(action_kwargs_list, list):
        action_kwargs = action_kwargs_list[trial]
      else:
        action_kwargs = action_kwargs_list

      state_history, result, info = self.simulate_one_trajectory(
          T_rollout=T_rollout, end_criterion=end_criterion,
          reset_kwargs=reset_kwargs, action_kwargs=action_kwargs,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback, **kwargs
      )
      trajectories.append(state_history)
      results[trial] = result
      length[trial] = len(state_history)
      info_list.append(info)
    if return_info:
      return trajectories, results, length, info_list
    else:
      return trajectories, results, length
