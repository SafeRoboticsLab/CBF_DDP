from typing import Dict, Tuple, Optional, Union
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from gym import spaces

from jax import numpy as jnp

from .bicycle5d_margin import BicycleReachAvoid5DMargin
from simulators.base_single_env import BaseSingleEnv


class CarSingle5DEnv(BaseSingleEnv):
  # region: init
  def __init__(self, config_env, config_agent, config_constraint) -> None:
    super().__init__(config_env, config_agent)

    self.track_width_right = config_env.TRACK_WIDTH_RIGHT
    self.track_width_left = config_env.TRACK_WIDTH_LEFT
    self.track_len = config_env.TRACK_LEN

    # Constructs the cost and constraint. Assume the same constraint.
    config_constraint.EGO_RADIUS = config_agent.EGO_RADIUS
    config_constraint.WHEELBASE = config_agent.WHEELBASE
    config_constraint.TRACK_WIDTH_LEFT = config_env.TRACK_WIDTH_LEFT
    config_constraint.TRACK_WIDTH_RIGHT = config_env.TRACK_WIDTH_RIGHT
    config_constraint.OBS_SPEC = config_env.OBS_SPEC
    config_constraint.OBSC_TYPE = config_env.OBSC_TYPE
    self.obsc_type = config_env.OBSC_TYPE 
    
    self.cost_type = getattr(config_constraint, "COST_TYPE", "Lagrange")
    if self.cost_type=="Reachavoid":
      self.cost = BicycleReachAvoid5DMargin(config_constraint, self.agent.dyn)
    elif self.cost_type=="Reachability":
      self.cost = BicycleReachAvoid5DMargin(config_constraint, self.agent.dyn)

    self.g_x_fail = config_env.G_X_FAIL

    # Visualization.
    x_min, y_min = 0, -1*config_constraint.TRACK_WIDTH_RIGHT
    x_max, y_max = self.track_len, config_constraint.TRACK_WIDTH_LEFT
    self.visual_bounds = np.array([[x_min, x_max], [y_min, y_max]])
    x_eps = (x_max-x_min) * 0.00
    y_eps = (y_max-y_min) * 0.00
    self.visual_extent = np.array([
        self.visual_bounds[0, 0] - x_eps, self.visual_bounds[0, 1] + x_eps,
        self.visual_bounds[1, 0] - y_eps, self.visual_bounds[1, 1] + y_eps
    ])
    self.obs_vertices_list = []

    if self.obsc_type=='circle':
      for circle_spec in self.cost.constraint.obs_spec:
        self.obs_vertices_list.append(circle_spec)

    # Initializes.
    self.reset_rej_sampling = getattr(config_env, "RESET_REJ_SAMPLING", True)
    self.build_obs_rst_space(config_env, config_agent, config_constraint)
    self.seed(config_env.SEED)
    self.reset()

  # endregion
  def get_constraints(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Dict:
    states_all, controls_all = self._reshape(state, action, state_nxt)
    states_all = jnp.array(states_all)
    controls_all = jnp.array(controls_all)
    cons_dict: Dict = self.cost.constraint.get_cost_dict(
        states_all, controls_all
    )
    for k, v in cons_dict.items():
      cons_dict[k] = np.asarray(v).reshape(-1, v.size)
    return cons_dict
  
  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray,
      constraints: Optional[dict] = None
  ) -> float:
    states_all, controls_all = self._reshape(state, action, state_nxt)
    states_all = jnp.array(states_all)
    controls_all = jnp.array(controls_all)
    cost = float(
        jnp.sum(
            self.cost.constraint.get_stage_margin(
                states_all, controls_all
            )
        )
    )
    return cost
  
  def get_target_margin(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> float:
    states_all, controls_all = self._reshape(state, action, state_nxt)
    states_all = jnp.array(states_all)
    controls_all = jnp.array(controls_all)
    cost = float(
        jnp.sum(
            self.cost.get_target_stage_margin(
                states_all, controls_all
            )
        )
    )
    return cost
  
  def get_done_and_info(
      self, state: np.ndarray, constraints: Dict,
      targets: Optional[Dict] = None, final_only: bool = True,
      end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    """
    Gets the done flag and a dictionary to provide additional information of
    the step function given current state, current action, next state,
    constraints, and targets.

    Args:
        constraints (Dict): each (key, value) pair is the name and value of a
            constraint function.
        targets (Dict): each (key, value) pair is the name and value of a
            target margin function.

    Returns:
        bool: True if the episode ends.
        Dict: additional information of the step, such as target margin and
            safety margin used in reachability analysis.
    """
    self.min_velocity = 0.0
    if end_criterion is None:
      end_criterion = self.end_criterion

    done = False
    done_type = "not_raised"
    if self.cnt >= self.timeout:
      done = True
      done_type = "timeout"
    if self.track_len is not None:
      if state[0] > self.track_len:
        done = True
        done_type = "leave_track with no failure"

    if state[2]<=self.min_velocity:
        done = True
        done_type = "safe stop"

    # Retrieves constraints / traget values.
    constraint_values = None
    for key, value in constraints.items():
      if constraint_values is None:
        num_pts = value.shape[1]
        constraint_values = value
      else:
        assert num_pts == value.shape[1], (
            "The length of constraint ({}) do not match".format(key)
        )
        constraint_values = np.concatenate((constraint_values, value), axis=0)
    g_x_list = np.min(constraint_values, axis=0)

    if targets is not None:
      target_values = np.empty((0, constraint_values.shape[1]))
      for key, value in targets.items():
        assert num_pts == value.shape[1], (
            "The length of target ({}) do not match".format(key)
        )
        target_values = np.concatenate((target_values, value), axis=0)
      l_x_list = np.max(target_values, axis=0)
    else:
      l_x_list = np.full((num_pts,), fill_value=np.inf)

    # Gets info.
    if final_only:
      g_x = g_x_list[-1]
      l_x = l_x_list[-1]
      binary_cost = 1. if g_x < 0. else 0.
    else:
      g_x = g_x_list
      l_x = l_x_list
      binary_cost = 1. if np.any(g_x < 0.) else 0.

    # Gets done flag
    if end_criterion == 'failure':
      if final_only:
        failure = np.any(constraint_values[:, -1] < 0.)
      else:
        failure = np.any(constraint_values < 0.)
      if failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail
    elif end_criterion == 'timeout':
      pass
    else:
      raise ValueError("End criterion not supported!")

    # Gets info
    info = {
        "done_type": done_type,
        "g_x": g_x,
        "l_x": l_x,
        "binary_cost": binary_cost
    }
    info['constraints'] = constraints
    return done, info

   # region: gym
  def build_obs_rst_space(self, config_env, config_agent, config_constraint):
    # Reset Sample Space. Note that the first two dimension is in the local
    # frame and it needs to call track.local2global() to get the (x, y)
    # position in the global frame.
    reset_space = np.array(config_env.RESET_SPACE, dtype=np.float32)
    self.reset_sample_space = spaces.Box(
        low=reset_space[:, 0], high=reset_space[:, 1]
    )

    # Observation space.
    x_min, y_min = 0, -1.2
    x_max, y_max = 6.0, -1.2
    
    low = np.zeros((self.state_dim,))
    low[0] = x_min
    low[1] = y_min
    low[2] = 0
    low[3] = -np.pi
    low[4] = -1
    #low[5] = -1
    high = np.zeros((self.state_dim,))
    high[0] = x_max
    high[1] = y_max
    high[2] = 1
    high[3] = np.pi
    high[4] = 1
    #high[5] = -1

    self.observation_space = spaces.Box(
        low=np.float32(low), high=np.float32(high)
    )
    self.obs_dim = self.observation_space.low.shape[0]

  # endregion
  def reset(
      self, state: Optional[np.ndarray] = None,
      **kwargs
  ) -> Union[np.ndarray, np.ndarray]:
    """
    Resets the environment and returns the new state.

    Args:
        state (np.ndarray, optional): reset to this state if provided. Defaults
            to None.
        cast_torch (bool, optional): cast state to torch if True. Defaults to
            False.

    Returns:
        np.ndarray: the new state of the shape (dim_x, ).
    """
    super().reset()
    if state is None:
      reset_flag = True
      while reset_flag:
        state = self.reset_sample_space.sample()
        if self.reset_rej_sampling:
          ctrl = jnp.zeros((self.action_dim, 1))
          state_jnp = jnp.array(state[:, np.newaxis])
          cons = self.cost.get_mapped_margin(
              state_jnp, ctrl
          )[0]
          reset_flag = cons > 0.
        else:
          reset_flag = False
    self.state = state.copy()

    obs = self.get_obs(state)
    return obs

  def get_obs(self, state: np.ndarray) -> np.ndarray:
    """Gets the observation given the state.

    Args:
        state (np.ndarray): state of the shape (dim_x, ) or  (dim_x, N).

    Returns:
        np.ndarray: observation. It can be the state or uses cos theta and
            sin theta to represent yaw.
    """
    assert state.shape[0] == self.state_dim, ("State shape is incorrect!")
    return state

  def get_samples(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gets state samples for value function plotting.

    Args:
        nx (int): the number of points along x-axis.
        ny (int): the number of points along y-axis.

    Returns:
        np.ndarray: a list of x-position.
        np.ndarray: a list of y-position.
    """
    xs = np.linspace(self.visual_bounds[0, 0], self.visual_bounds[0, 1], nx)
    ys = np.linspace(self.visual_bounds[1, 0], self.visual_bounds[1, 1], ny)
    return xs, ys

  # endregion

  # region: utils

  def render(
      self, ax: Optional[matplotlib.axes.Axes] = None, c_track: str = 'k',
      c_obs: str = 'r', c_ego: str = 'b', s: float = 12
  ):
    """Visualizes the current environment.

    Args:
        ax (Optional[matplotlib.axes.Axes], optional): the axes of matplotlib
            to plot if provided. Otherwise, we use the current axes. Defaults
            to None.
        c_track (str, optional): the color of the track. Defaults to 'k'.
        c_obs (str, optional): the color of the obstacles. Defaults to 'r'.
        c_ego (str, optional): the color of the ego agent. Defaults to 'b'.
        s (float, optional): the size of the ego agent point. Defaults to 12.
    """
    if ax is None:
      ax = plt.gca()
    self.render_footprint(ax=ax, state=self.state, c=c_ego)
    self.render_obs(ax=ax, c=c_obs)
    ax.axis(self.visual_extent)
    ax.set_aspect('equal')

  def render_footprint(
      self, ax, state: np.ndarray, c: str = 'b', s: float = 12,
      lw: float = 1.5, alpha: float = 1.
  ):
    ax.scatter(state[0], state[1], c=c, s=s)
    ego = self.agent.footprint
    ego.set_center(state[0:2])
    ego.plot(ax, color=c, lw=lw, alpha=alpha)

  def render_obs(self, ax, c: str = 'r'):
    if self.cost.constraint.obsc_type=='box':
      for vertices in self.obs_vertices_list:
        for i in range(4):
          if i == 3:
            ax.plot(vertices[[3, 0], 0], vertices[[3, 0], 1], c=c)
          else:
            ax.plot(vertices[i:i + 2, 0], vertices[i:i + 2, 1], c=c)
    elif self.cost.constraint.obsc_type=='circle':
      for vertices in self.obs_vertices_list:
        obs_circle = plt.Circle([vertices[0], vertices[1]], vertices[2], alpha=0.4, color=c)
        ax.add_patch(obs_circle)


  def render_state_cost_map(
      self, ax, nx: int, ny: int, vel: float, yaw: float, delta: float,
      vmin: float = 0., vmax: float = 20., cmap: str = 'seismic',
      alpha: float = 0.5, cost_type: str = 'cost'
  ):
    xmin, xmax = self.visual_bounds[0]
    ymin, ymax = self.visual_bounds[1]
    state = np.zeros((5, nx * ny))
    offset_xs = np.linspace(xmin, xmax, nx)
    offset_ys = np.linspace(ymin, ymax, ny)
    offset_xv, offset_yv = np.meshgrid(offset_xs, offset_ys, indexing='ij')
    offset = np.concatenate(
        (offset_xv[..., np.newaxis], offset_yv[..., np.newaxis]), axis=-1
    )
    state[:2, :] = np.array(offset.reshape(-1, 2)).T
    state[2, :] = vel
    state[3, :] = yaw
    state[4, :] = delta
    ctrl = np.zeros((self.action_dim, nx * ny))

    state = jnp.array(state)
    ctrl = jnp.array(ctrl)

    if cost_type == "cost":
      cost = self.cost
    else:
      assert cost_type == "constraint"
      cost = self.cost
    v = cost.get_stage_margin(state, ctrl).reshape(nx, ny)
    ax.imshow(
        v.T, interpolation='none', extent=[xmin, xmax, ymin, ymax],
        origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, zorder=-1, alpha=alpha
    )

  def _reshape(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenates states with state_nxt and appends dummy control after action.

    Args:
        state (np.ndarray): current states of the shape (dim_x, N-1).
        action (np.ndarray): current actions of the shape (dim_u, N-1).
        state_nxt (np.ndarray): next state or final state of the shape
            (dim_x, ).

    Returns:
        np.ndarray: states with the final state.
        np.ndarray: action with the dummy control.
    """
    if state.ndim == 1:
      state = state[:, np.newaxis]
    if state_nxt.ndim == 1:
      state_nxt = state_nxt[:, np.newaxis]
    ctrl = action
    if ctrl.ndim == 1:
      ctrl = ctrl[:, np.newaxis]
    assert state.shape[1] == ctrl.shape[1], (
        "The length of states ({}) and ".format(state.shape[1]),
        "the length of controls ({}) don't match!".format(ctrl.shape[1])
    )
    assert state_nxt.shape[1] == 1, "state_nxt should consist only 1 state!"

    states_with_final = np.concatenate((state, state_nxt), axis=1)
    controls_with_final = np.concatenate((ctrl, np.zeros((ctrl.shape[0], 1))),
                                         axis=1)
    return states_with_final, controls_with_final

  def report(self):
    if self.track_len is not None:
      print("Straight road, circle footprint, circle obstacles!")
    else:
      print("road from file, circle footprint, circle obstacles!")

  # endregion
