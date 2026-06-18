from abc import ABC, abstractmethod
from typing import Tuple, Optional
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp


class BaseMargin(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        raise NotImplementedError

    @abstractmethod
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        raise NotImplementedError

    @partial(jax.jit, static_argnames='self')
    def get_mapped_margin(self, state: DeviceArray,
                          ctrl: DeviceArray) -> DeviceArray:
        return jax.vmap(self.get_stage_margin, in_axes=(1, 1))(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_mapped_target_margin(
            self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        return jax.vmap(self.get_target_stage_margin,
                        in_axes=(1, 1))(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_mapped_target_margin_with_derivative(
            self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        return jax.vmap(self.get_target_stage_margin_with_derivative, in_axes=(
            1, 1), out_axes=((0), (1), (2), (1), (2)))(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_traj_cost(self, state: DeviceArray, ctrl: DeviceArray) -> float:
        costs = jax.vmap(self.get_stage_margin, in_axes=(1, 1))(state, ctrl)
        return jnp.sum(costs).astype(float)

    @partial(jax.jit, static_argnames='self')
    def get_cx(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cx = jax.jacfwd(self.get_stage_margin, argnums=0)
        return jax.vmap(_cx, in_axes=(1, 1), out_axes=1)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cu(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cu = jax.jacfwd(self.get_stage_margin, argnums=1)
        return jax.vmap(_cu, in_axes=(1, 1), out_axes=1)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cxx(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cxx = jax.jacfwd(
            jax.jacrev(
                self.get_stage_margin,
                argnums=0),
            argnums=0)
        return jax.vmap(_cxx, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cuu(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cuu = jax.jacfwd(
            jax.jacrev(
                self.get_stage_margin,
                argnums=1),
            argnums=1)
        return jax.vmap(_cuu, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cux(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cux = jax.jacfwd(
            jax.jacrev(
                self.get_stage_margin,
                argnums=1),
            argnums=0)
        return jax.vmap(_cux, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cxu(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        return self.get_cux(state, ctrl).T

    @partial(jax.jit, static_argnames='self')
    def get_cx_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cx = jax.jacfwd(self.get_target_stage_margin, argnums=0)
        return jax.vmap(_cx, in_axes=(1, 1), out_axes=1)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cu_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cu = jax.jacfwd(self.get_target_stage_margin, argnums=1)
        return jax.vmap(_cu, in_axes=(1, 1), out_axes=1)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cxx_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cxx = jax.jacfwd(
            jax.jacfwd(
                self.get_target_stage_margin,
                argnums=0),
            argnums=0)
        return jax.vmap(_cxx, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cuu_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cuu = jax.jacfwd(
            jax.jacrev(
                self.get_target_stage_margin,
                argnums=1),
            argnums=1)
        return jax.vmap(_cuu, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cux_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        _cux = jax.jacfwd(
            jax.jacrev(
                self.get_target_stage_margin,
                argnums=1),
            argnums=0)
        return jax.vmap(_cux, in_axes=(1, 1), out_axes=2)(state, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_cxu_t(self, state: DeviceArray, ctrl: DeviceArray) -> DeviceArray:
        return self.get_cux_t(state, ctrl).T

    @partial(jax.jit, static_argnames='self')
    def get_derivatives(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return (
            self.get_cx(state, ctrl),
            self.get_cu(state, ctrl),
            self.get_cxx(state, ctrl),
            self.get_cuu(state, ctrl),
            self.get_cux(state, ctrl),
        )

    @partial(jax.jit, static_argnames='self')
    def get_derivatives_target(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return (
            self.get_cx_t(state, ctrl),
            self.get_cu_t(state, ctrl),
            self.get_cxx_t(state, ctrl),
            self.get_cuu_t(state, ctrl),
            self.get_cux_t(state, ctrl),
        )


class SoftBarrierEnvelope(BaseMargin):
    def __init__(
        self, clip_min: Optional[float], clip_max: Optional[float], q1: float,
        q2: float, margin: BaseMargin
    ):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.q1 = q1
        self.q2 = q2
        self.margin = margin

    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        _cost = self.margin.get_stage_margin(state, ctrl)
        return self.q1 * jnp.exp(
            self.q2 * jnp.clip(a=_cost, a_min=self.clip_min,
                               a_max=self.clip_max)
        )

    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)
