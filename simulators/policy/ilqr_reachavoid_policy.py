from typing import Tuple, Optional, Dict
import time
import numpy as np
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray
#from jax.experimental import checkify
from functools import partial

from .ilqr_policy import iLQR


class iLQRReachAvoid(iLQR):

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None,**kwargs
  ) -> np.ndarray:
    status = 0
    self.tol = 1e-5

    if controls is None:
      controls = np.zeros((self.dim_u, self.N))
      # Some non-zero initialization
      controls[0, :] = self.dyn.ctrl_space[0, 0]
      controls = jnp.array(controls)
    else:
      assert controls.shape[1] == self.N
      controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    states, controls = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls
    )
    
    failure_margins = self.cost.constraint.get_mapped_margin(
        states, controls
    )

    # Target cost derivatives are manually computed for more well-behaved backpropagation
    #target_margins = self.cost.get_mapped_target_margin(states, controls)
    target_margins, c_x_t, c_xx_t, c_u_t, c_uu_t = self.cost.get_mapped_target_margin_with_derivative(states, controls)

    is_inside_target = (target_margins[0]>0)
    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(states, controls)
    critical, reachavoid__margin = self.get_critical_points(failure_margins, target_margins)
        
    J = (reachavoid__margin + jnp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()

    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          states, controls
      )

      #c_x_t, c_u_t, c_xx_t, c_uu_t, c_ux_t = self.cost.get_derivatives_target(
      #    states, controls
      #)

      fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
      V_x, V_xx, k_open_loop, K_closed_loop, _, _ = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, 
          c_x_t=c_x_t, c_u_t=c_u_t, c_xx_t=c_xx_t, c_uu_t=c_uu_t, c_ux_t=None, fx=fx, fu=fu,
          critical=critical
      )
      
      # Choose the best alpha scaling using appropriate line search methods
      #alpha_chosen = self.baseline_line_search( states, controls, K_closed_loop, k_open_loop, J)
      #alpha_chosen = self.armijo_line_search( states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical, 
      #                                       J=J, c_u=c_u)
      alpha_chosen = self.trust_region_search_conservative( states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical, 
                                             J=J, c_x=c_x, c_xx=c_xx)

      #states, controls, J_new, critical, failure_margins, target_margins, reachavoid_margin, _, _, _, _ = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha_chosen)        
      states, controls, J_new, critical, failure_margins, target_margins, reachavoid_margin, c_x_t, c_xx_t, c_u_t, c_uu_t = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha_chosen) 
      if (np.abs((J-J_new) / J) < self.tol):  # Small improvement.
        status = 1
        if J_new>0:
          converged = True

      J = J_new
      
      if alpha_chosen<self.min_alpha:
          status = 2
          break
      # Terminates early if the objective improvement is negligible.
      if converged:
        break

    t_process = time.time() - time0
    states = np.asarray(states)
    controls = np.asarray(controls)
    K_closed_loop = np.asarray(K_closed_loop)
    k_open_loop = np.asarray(k_open_loop)
    solver_info = dict(
        states=states, controls=controls, reinit_controls=controls, t_process=t_process, 
        status=status, Vopt=J, marginopt=reachavoid_margin,
        grad_x=V_x, grad_xx=V_xx, B0=fu[:, :, 0], critical=critical, 
        is_inside_target=is_inside_target, K_closed_loop=K_closed_loop, k_open_loop=k_open_loop
    )

    return controls[:, 0], solver_info
  
  @partial(jax.jit, static_argnames='self')
  def baseline_line_search(self, states, controls, K_closed_loop, k_open_loop, J, beta=0.9):
    alpha = 1.0
    J_new = -jnp.inf

    @jax.jit
    def run_forward_pass(args):
      states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = args
      alpha = beta*alpha
      _, _, J_new, _, _, _, _ = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
      return states, controls, K_closed_loop, k_open_loop, alpha, J, J_new

    @jax.jit
    def check_continue(args):
      _, _, _, _, alpha, J, J_new = args
      return jnp.logical_and( alpha>self.min_alpha, J_new<J )
    
    states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = jax.lax.while_loop(check_continue, 
                                                                                       run_forward_pass, 
                                                                                       (states, controls, K_closed_loop, k_open_loop, alpha, J, J_new))

    return alpha

  @partial(jax.jit, static_argnames='self')
  def armijo_line_search( self, states, controls, Ks1, ks1, critical, J, c_u, alpha_init=1.0, beta=0.8):
    alpha = 1.0
    J_new = -jnp.inf
    deltat = 0
    t_star = jnp.where(critical!=0, size=self.N-1)[0][0]
    
    @jax.jit
    def run_forward_pass(args):
      states, controls, Ks1, ks1, alpha, J, t_star, J_new, deltat= args
      alpha = beta*alpha
      X, _, J_new, _, _, _, _ = self.forward_pass(nominal_states=states, nominal_controls=controls, 
                                                     K_closed_loop=Ks1, k_open_loop=ks1, alpha=alpha)      
      # Calculate gradient for armijo decrease condition
      delta_u =  Ks1[:, :, t_star] @ (X[:, t_star]  - states[:, t_star]) + ks1[:, t_star]

      grad_cost_u = c_u[:, t_star]

      deltat = grad_cost_u @ delta_u

      return  states, controls, Ks1, ks1, alpha, J, t_star, J_new, deltat

    @jax.jit
    def check_continue(args):
      _, _, _, _, alpha, J, _, J_new, deltat = args
      armijo_check = (J_new <=J + deltat*alpha )
      return jnp.logical_and( alpha>self.min_alpha, armijo_check )
    
    states, controls, Ks1, ks1, alpha, J, t_star, J_new, deltat = jax.lax.while_loop(check_continue, run_forward_pass, (states, controls,
                                                                                                                            Ks1, ks1, alpha, J, t_star, J_new, deltat))
    return alpha
  
  @partial(jax.jit, static_argnames='self')
  def trust_region_search_conservative( self, states, controls, Ks1, ks1, J, critical, 
                         c_x, c_xx, alpha_init=1.0, beta=0.8):
    alpha = alpha_init
    J_new = -jnp.inf
    t_star = jnp.where(critical!=0, size=self.N-1)[0][0]

    self.margin = 2
    traj_diff = 0.2
    cost_error = 1.0
    old_cost_error = 2.0
    rho = 0.5

    @jax.jit
    def decrease_margin(args):
      self.margin = 0.75*self.margin
      return args
    
    @jax.jit
    def increase_margin(args):
      self.margin = 1.4*self.margin
      return args

    @jax.jit
    def fix_margin(args):
      return args
    
    @jax.jit
    def increase_or_fix_margin(args):
      cost_error, old_cost_error, traj_diff, rho = args
      _, _ = jax.lax.cond(jnp.logical_and( jnp.abs(traj_diff - self.margin)<0.01, rho>0.75), increase_margin, 
                                                fix_margin, (cost_error, old_cost_error))     
      return cost_error, old_cost_error, traj_diff, rho

    @jax.jit
    def run_forward_pass(args):
      states, controls, Ks1, ks1, alpha, J, t_star, J_new, traj_diff, cost_error, old_cost_error, rho = args
      alpha = beta*alpha
      X, _, J_new, _, _, _, _, _, _, _, _ = self.forward_pass(nominal_states=states, nominal_controls=controls, 
                                                     K_closed_loop=Ks1, k_open_loop=ks1, alpha=alpha)      

      traj_diff = jnp.max(jnp.array([jnp.linalg.norm( x_new - x_old )
                          for x_new, x_old in zip(X[:2, :], states[:2, :])]))
      
      x_diff = X[:, t_star] - states[:, t_star]

      delta_cost_quadratic_approx = 0.5* (x_diff @ c_xx[:, :, t_star] + 2*c_x[:, t_star]) @ x_diff
      delta_cost_actual = J_new - J
      old_cost_error = jnp.abs(cost_error)
      cost_error = jnp.abs( delta_cost_quadratic_approx - delta_cost_actual )
      rho = delta_cost_actual/delta_cost_quadratic_approx
      return  states, controls, Ks1, ks1,alpha, J, t_star, J_new, traj_diff, cost_error, old_cost_error, rho

    @jax.jit
    def check_continue(args):
      _, _, _, _, alpha, J, _, J_new, traj_diff, cost_error, old_cost_error, rho = args
      trust_region_violation = ( traj_diff>self.margin )
      improvement_violation = (J_new < J)

      cost_error, old_cost_error, traj_diff, rho = jax.lax.cond((rho<=0.25), decrease_margin, 
                                                increase_or_fix_margin, (cost_error, old_cost_error, traj_diff, rho))      
      return jnp.logical_and( alpha>self.min_alpha, jnp.logical_or(trust_region_violation , improvement_violation))
    
    states, controls, Ks1, ks1, alpha, J, t_star, J_new, traj_diff, cost_error, _, _ = (
      jax.lax.while_loop(check_continue, run_forward_pass, (states, controls,
                                                          Ks1, ks1, alpha, J, t_star, J_new, 
                                                          traj_diff, cost_error, old_cost_error, rho)))
    #print("Margin", self.margin)
    return alpha

  
  @partial(jax.jit, static_argnames='self')
  def get_critical_points(
      self, failure_margins: DeviceArray, target_margins:DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    # Avoid cost is critical
    @jax.jit
    def failure_func(args):
      idx, critical, failure_margin, _, _ = args
      critical = critical.at[idx].set(1)
      return critical, failure_margin

    # Avoid cost is not critical
    @jax.jit
    def target_propagate_func(args):
      _, _, _, target_margin, reachavoid_margin = args
      return jax.lax.cond(target_margin > reachavoid_margin, target_func, propagate_func, args)  

    # Reach cost is critical
    @jax.jit
    def target_func(args):
      idx, critical, _, target_margin, _ = args
      critical = critical.at[idx].set(2)
      return critical, target_margin

    # Propagating the cost is critical
    @jax.jit
    def propagate_func(args):
      idx, critical, _, _, reachavoid_margin = args
      critical = critical.at[idx].set(0)
      return critical, reachavoid_margin

    @jax.jit
    def critical_pt(i, _carry):
      idx = self.N - 1 - i
      critical, reachavoid_margin = _carry

      failure_margin = failure_margins[idx]
      target_margin = target_margins[idx]
      critical, reachavoid_margin = jax.lax.cond(
          ( (failure_margin < reachavoid_margin) | (failure_margin < target_margin) ), 
          failure_func, target_propagate_func,
          (idx, critical, failure_margin, target_margin, reachavoid_margin)
      )

      return critical, reachavoid_margin

    critical = jnp.zeros(shape=(self.N,), dtype=int)

    reachavoid_margin = 0.
    critical, reachavoid_margin = jax.lax.cond(target_margins[self.N - 1] < failure_margins[self.N - 1], 
                                               target_func, failure_func, 
                  (self.N - 1, critical, failure_margins[self.N - 1], 
                   target_margins[self.N - 1], reachavoid_margin)
    )

    critical, reachavoid_margin = jax.lax.fori_loop(
        1, self.N, critical_pt, (critical, reachavoid_margin)
    )  # backward until timestep 1

    return critical, reachavoid_margin

  @partial(jax.jit, static_argnames='self')
  def forward_pass(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
             DeviceArray, DeviceArray]:
    X, U = self.rollout(
        nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )

    # J = self.cost.get_traj_cost(X, U, closest_pt, slope, theta)
    #! hacky
    failure_margins = self.cost.constraint.get_mapped_margin(X, U)
    #target_margins = self.cost.get_mapped_target_margin(X, U)
    target_margins, c_x_t, c_xx_t, c_u_t, c_uu_t = self.cost.get_mapped_target_margin_with_derivative(X, U)

    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(X, U)

    critical, reachavoid_margin = self.get_critical_points(failure_margins, target_margins)

    J = (reachavoid_margin + jnp.sum(ctrl_costs)).astype(float)
    
    #reachavoid_margin_comp = self.brute_force_critical_cost(np.array(failure_margins), np.array(target_margins))
    #assert jnp.abs(reachavoid_margin - reachavoid_margin_comp)==0
    #checked_margin = checkify.checkify(self.brute_force_critical_cost)
    #err, future_cost = checked_margin(failure_margins, target_margins, reachavoid_margin)
    #err.throw()

    return X, U, J, critical, failure_margins, target_margins, reachavoid_margin, c_x_t, c_xx_t, c_u_t, c_uu_t

  """
  def brute_force_critical_cost(self, state_costs, target_costs):
    critical_cost_arr = np.zeros((self.N, ))
    for iters in range(self.N):
      min_avoid_cost = np.min(state_costs[0:iters+1])
      critical_cost_arr[iters] = np.minimum(min_avoid_cost, target_costs[iters])
    return np.max(critical_cost_arr)

  #@partial(jax.jit, static_argnames='self')
  def brute_force_critical_cost(self, failure_margins, target_margins):
    critical_cost_arr = jnp.zeros((self.N, ))

    def forward_looper(idx, args):
      critical_cost_arr, min_failure_margin = args
      min_failure_margin = jnp.minimum(min_failure_margin, failure_margins[idx])
      critical_cost_arr = critical_cost_arr.at[idx].set(jnp.minimum(min_failure_margin, target_margins[idx]))
      return critical_cost_arr, min_failure_margin
    
    critical_cost_arr, _ = jax.lax.fori_loop(0, self.N, forward_looper, (critical_cost_arr, failure_margins[0]))
    future_cost = jnp.max(critical_cost_arr)
    #checkify.check( future_cost - compare_cost==0, "Reach-avoid cost error")

    return future_cost
  """

  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, 
      c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, c_x_t: DeviceArray, c_u_t: DeviceArray, 
      c_xx_t: DeviceArray, c_uu_t: DeviceArray, c_ux_t: DeviceArray, fx: DeviceArray, fu: DeviceArray,
      critical: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray]:
    """
    Jitted backward pass looped computation.

    Args:
        c_x (DeviceArray): (dim_x, N)
        c_u (DeviceArray): (dim_u, N)
        c_xx (DeviceArray): (dim_x, dim_x, N)
        c_uu (DeviceArray): (dim_u, dim_u, N)
        c_ux (DeviceArray): (dim_u, dim_x, N)
        fx (DeviceArray): (dim_x, dim_x, N-1)
        fu (DeviceArray): (dim_x, dim_u, N-1)

    Returns:
        Vx (DeviceArray): gain matrices (dim_x, 1)
        Vxx (DeviceArray): gain vectors (dim_x, dim_x, 1)
        Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
        ks (DeviceArray): gain vectors (dim_u, N - 1)
        Vx_critical (DeviceArray): gain matrices (dim_x, 1)
        Vxx_critical (DeviceArray): gain vectors (dim_x, dim_x, 1)
    """

    @jax.jit
    def failure_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)
 
      return jnp.array( c_x[:, idx] ), jnp.array( c_xx[:, :, idx] ), ks, Ks, jnp.array( c_x[:, idx] ), jnp.array( c_xx[:, :, idx] )

    @jax.jit
    def target_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u_t[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu_t[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)
 
      return jnp.array( c_x_t[:, idx] ), jnp.array( c_xx_t[:, :, idx] ), ks, Ks, jnp.array( c_x_t[:, idx] ), jnp.array( c_xx_t[:, :, idx] )

    @jax.jit
    def propagate_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

      Q_x = fx[:, :, idx].T @ V_x
      Q_xx = fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)

      V_x = Q_x + Q_ux.T @ ks[:, idx]
      V_xx = Q_xx + Q_ux.T @ Ks[:, :, idx]

      return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical

    @jax.jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical = _carry
      idx = self.N - 2 - i

      V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = jax.lax.switch(
          critical[idx], [propagate_backward_func, failure_backward_func, target_backward_func], 
          (idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical)
      )

      return V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical

    @jax.jit
    def failure_final_func(args):
      c_x, c_xx, _, _ = args
      return jnp.array( c_x[:, self.N - 1] ), jnp.array( c_xx[:, :, self.N - 1] )

    @jax.jit
    def target_final_func(args):
      _, _, c_x_t, c_xx_t = args
      return jnp.array( c_x_t[:, self.N - 1] ), jnp.array( c_xx_t[:, :, self.N - 1] )

    # Initializes.
    Ks = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
    ks = jnp.zeros((self.dim_u, self.N - 1))
    
    V_x_critical = jnp.zeros((self.dim_x, ))
    V_xx_critical = jnp.zeros((self.dim_x, self.dim_x, ))

    reg_mat = self.eps * jnp.eye(self.dim_x)

    # If critical is 2 choose target - hacky!!
    V_x, V_xx = jax.lax.cond(critical[self.N -1]==1, failure_final_func, target_final_func, (c_x, c_xx, c_x_t, c_xx_t))

    V_x, V_xx, ks, Ks, _, V_x_critical, V_xx_critical = jax.lax.fori_loop(
        0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical)
    )

    return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical