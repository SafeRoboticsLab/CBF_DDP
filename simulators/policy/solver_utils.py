import cvxpy as cp
from cvxpy.error import SolverError
import numpy as np

import math

import jax
from jax import numpy as jnp

@jax.jit
def barrier_filter_linear(grad_x, B0, c):
  B0 = B0[:, :, 0]
  p = grad_x.T @ B0
  return -c*p/(jnp.dot(p, p))

def barrier_filter_quadratic(P, p, c, initialize): 
    def is_neg_def(x):
      # Check if a matrix is PSD
      return np.all( np.real( np.linalg.eigvals(x) ) < 0) 

    #CVX faces numerical difficulties otherwise
    check_nd = is_neg_def(P)

    # Check if P is PD
    if(check_nd):
      u = cp.Variable((2))
      u.value = np.array( initialize )
      P = np.array(P)
      p = np.array(p)
      
      prob = cp.Problem( cp.Minimize(1.0*cp.square(u[0]) + 1.0*cp.square(u[1])),
                    [ cp.quad_form(u, P) + p.T@u + c >=0 ] )
      try:
        prob.solve(verbose=False, warm_start=True)
      except SolverError:
        pass
    
    if(not check_nd or u[0] is None or prob.status not in ["optimal","optimal_inaccurate"]):
      u = cp.Variable((2))
      u.value = np.array( initialize )
      p = np.array(p)
      prob = cp.Problem( cp.Minimize(1.0*cp.square(u[0]) + 1.0*cp.square(u[1])),
                      [ p @ u + c >= 0] )
      try:
        prob.solve(verbose=False, warm_start=True)
      except SolverError:
        pass
      
    if prob.status not in ["optimal","optimal_inaccurate"] or u[0] is None:
      return np.array([0., 0.]) 
    return np.array([u[0].value, u[1].value])

# Unused as we started using iLQR task policy
def bicycle_linear_task_policy( run_env_obs ):
    lookahead_distance = 4.0
    direction_waypoint =  math.atan2(-run_env_obs[1], lookahead_distance)    

    if run_env_obs.size==5:
      control_task = np.zeros((2, ))
      control_task[0] = -1.0*(run_env_obs[2] - 0.9) 
      # Use only unwrapped yaw phase for this subtraction
      alpha = (run_env_obs[3] - direction_waypoint)

      if np.abs(alpha)<0.1:
        control_task[1] = -1.0*run_env_obs[4] 
      else:    
        control_task[1] = -1.2*alpha - 0.9*run_env_obs[4]

    return control_task