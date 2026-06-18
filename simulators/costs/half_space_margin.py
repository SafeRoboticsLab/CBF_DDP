import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as DeviceArray
from functools import partial

from .base_margin import BaseMargin


class UpperHalfMargin(BaseMargin):
    """
    c = val - x[dim]
    """

    def __init__(self, value: float, buffer: float, dim: int):
        super().__init__()
        self.dim = dim
        self.value = value
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        c = self.value - state[self.dim]
        c = c - self.buffer
        return c

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


class LowerHalfMargin(BaseMargin):
    """
    c = x[dim] - val
    """

    def __init__(self, value: float, buffer: float, dim: int):
        super().__init__()
        self.dim = dim
        self.value = value
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        c = state[self.dim] - self.value
        c = c - self.buffer
        return c

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)
