import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as DeviceArray
from functools import partial

from .base_margin import BaseMargin


class CircleObsMargin(BaseMargin):
    """
    We want s[i] < lb[i] or s[i] > ub[i].
    """

    def __init__(
        self, circle_spec=np.ndarray,
        buffer: float = 0.
    ):
        super().__init__()
        self.center = jnp.array(circle_spec[0:2])
        self.radius = circle_spec[2]
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        # signed distance to the box, positive inside
        circ_distance = jnp.sqrt(
            (state[0] - self.center[0])**2 + (state[1] - self.center[1])**2) - self.radius
        return circ_distance - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


class BoxObsMargin(BaseMargin):
    """
    We want s[i] < lb[i] or s[i] > ub[i].
    """

    def __init__(
        self, box_spec=np.ndarray,
        buffer: float = 0.
    ):
        super().__init__()
        self.ub = jnp.array(
            [box_spec[0] + box_spec[3], box_spec[1] + box_spec[4]])
        self.lb = jnp.array(
            [box_spec[0] - box_spec[3], box_spec[1] - box_spec[4]])
        self.n_dims = len(self.ub)
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        # signed distance to the box, positive inside
        sgn_dist = jnp.minimum(state[0] - self.lb[0], self.ub[0] - state[0])
        for i in range(1, self.n_dims):
            sgn_dist = jnp.minimum(sgn_dist, self.ub[i] - state[i])
            sgn_dist = jnp.minimum(sgn_dist, state[i] - self.lb[i])
        return -sgn_dist - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)
