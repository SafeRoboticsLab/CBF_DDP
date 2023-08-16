from __future__ import annotations
import sys
from typing import TypeVar, TypedDict, List, Optional, Union, Tuple, Dict
import numpy as np
import pickle

from gym import spaces

def save_obj(obj, filename, protocol=None):
  if protocol is None:
    protocol = pickle.HIGHEST_PROTOCOL
  with open(filename + '.pkl', 'wb') as f:
    pickle.dump(obj, f, protocol=protocol)


def load_obj(filename):
  with open(filename + '.pkl', 'rb') as f:
    return pickle.load(f)


class PrintLogger(object):
  """
  This class redirects print statements to both console and a file.
  """

  def __init__(self, log_file):
    self.terminal = sys.stdout
    print('STDOUT will be forked to %s' % log_file)
    self.log_file = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log_file.write(message)
    self.log_file.flush()

  def flush(self):
    # this flush method is needed for python 3 compatibility.
    # this handles the flush command by doing nothing.
    # you might want to specify some extra behavior here.
    pass


# Type Hints
class ActionZS(TypedDict):
  ctrl: np.ndarray
  dstb: np.ndarray


GenericAction = TypeVar(
    'GenericAction', np.ndarray, Dict[str, np.ndarray], ActionZS
)

GenericState = TypeVar(
    'GenericState', np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
)


# Observation.
def build_space(
    space_spec: np.ndarray, space_dim: Optional[tuple] = None
) -> spaces.Box:
  if space_spec.ndim == 2:  # e.g., state.
    obs_space = spaces.Box(low=space_spec[:, 0], high=space_spec[:, 1])
  elif space_spec.ndim == 4:  # e.g., RGB-D.
    obs_space = spaces.Box(
        low=space_spec[:, :, :, 0], high=space_spec[:, :, :, 1]
    )
  else:  # Each dimension shares the same min and max.
    assert space_spec.ndim == 1, "Unsupported space spec!"
    assert space_dim is not None, "Space dim is not provided"
    obs_space = spaces.Box(
        low=space_spec[0], high=space_spec[1], shape=space_dim
    )
  return obs_space


def concatenate_obs(observations: List[np.ndarray]) -> np.ndarray:
  base_shape = observations[0].shape[1:]
  flags = np.array([x.shape[1:] == base_shape for x in observations])
  assert np.all(flags), (
      "The obs. of each agent should be the same except the first dim!"
  )
  return np.concatenate(observations)


# Math Operator.
def barrier_function(
    q1: float, q2: float, cons: np.ndarray | float, cons_dot: np.ndarray,
    cons_min: Optional[float] = None, cons_max: Optional[float] = None
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
  clip = not (cons_min is None and cons_max is None)
  if clip:
    tmp = np.clip(q2 * cons, cons_min, cons_max)
  else:
    tmp = q2 * cons
  b = q1 * (np.exp(tmp))

  if isinstance(cons, np.ndarray):
    b = b.reshape(-1)
    assert b.shape[0] == cons_dot.shape[1], (
        "The shape of cons and cons_dot don't match!"
    )
    b_dot = np.einsum('n,an->an', q2 * b, cons_dot)
    b_ddot = np.einsum(
        'n,abn->abn', (q2**2) * b, np.einsum('an,bn->abn', cons_dot, cons_dot)
    )
  elif isinstance(cons, float):
    cons_dot = cons_dot.reshape(-1, 1)  # Transforms to column vector.
    b_dot = q2 * b * cons_dot
    b_ddot = (q2**2) * b * np.einsum('ik,jk->ij', cons_dot, cons_dot)
  else:
    raise TypeError("The type of cons is not supported!")
  return b_dot, b_ddot
