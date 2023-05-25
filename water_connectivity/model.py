import numpy as np
import h5py
import numba as nb
import os
from time import time
from pathlib import Path
import pde
from pde.tools import mpi
from pde.solvers import Controller
from pde.solvers.explicit_mpi import ExplicitMPISolver

from h5py_utils import write_field


class ModelPDE(pde.PDEBase):
    def __init__(
        self,
        bc,
        terrain,
        nu=10 / 3,
        eta=3.5,
        rho=0.95,
        gamma=50 / 3,
        delta_b=1 / 30,
        delta_w=10 / 3,
        delta_h=1e-2 / 3,
        a=33.33,
        q=0.05,
        f=0.1,
        p=0.5,
    ):
        self._nu = nu
        self._eta = eta
        self._rho = rho
        self._gamma = gamma
        self._delta_b = delta_b
        self._delta_w = delta_w
        self._delta_h = delta_h
        self._a = a
        self._q = q
        self._f = f
        self._p = p

        self._bc = bc
        self._terrain = terrain

    def _make_pde_rhs_numba(self, state):
        a = self._a
        q = self._q
        f = self._f
        eta = self._eta
        nu = self._nu
        rho = self._rho
        gamma = self._gamma
        delta_b = self._delta_b
        delta_w = self._delta_w
        delta_h = self._delta_h
        p = self._p

        laplace = state.grid.make_operator("laplace", bc=self._bc)
        grad = state.grid.make_operator("gradient", bc=self._bc)
        div = state.grid.make_operator("divergence", bc=self._bc)
        zeta = self._terrain.split_mpi()
        zeta = zeta.data

        @nb.jit
        def pde_rhs(state_data, t):
            # state
            b, w, h = state_data

            modified_terrain = h + zeta
            j = -2 * delta_h * h * grad(modified_terrain)

            rate = np.empty_like(state_data)
            # Calculate constants
            L2 = np.float_power(1 + eta * b, 2)
            gb = nu * w * L2
            gw = gamma * b * L2
            i = a * (b + q * f) / (b + q)

            # Calculate time derivatives
            rate[0] = gb * b * (1 - b) - b + delta_b * laplace(b)
            rate[1] = i * h - nu * (1 - rho * b) * w - gw * w + delta_w * laplace(w)
            rate[2] = p - i * h - div(j)

            return rate

        return pde_rhs

    def evolution_rate(self, state: pde.FieldBase, t: float = 0) -> pde.FieldBase:
        b, w, h = state

        # Calculate constants
        i = self._a * (b + self._q * self._f) / (b + self._q)
        l2 = np.float_power(1 + self._eta * b, 2)
        gb = self._nu * w * l2
        gw = self._gamma * b * l2

        zeta = self._terrain.split_mpi()
        modified_terrain = h + zeta
        j = -2 * self._delta_h * h * modified_terrain.gradient(self._bc)

        # Calculate time derivatives
        b_t = gb * b * (1 - b) - b + self._delta_b * b.laplace(self._bc)
        w_t = (
            i * h
            - self._nu * (1 - self._rho * b) * w
            - gw * w
            + self._delta_w * w.laplace(self._bc)
        )
        h_t = self._p - i * h - j.divergence(self._bc)

        return pde.FieldCollection([b_t, w_t, h_t])


def run_simulation(output: Path, n: int, tmax: int, L: int, percipitation: float, slope: float):
    # define constants for simulation
    dx = L / n
    dt = 0.5 * np.power(dx, 2) * 1e-1
    shape = (n, n)
    grid_range = [(0, L), (0, L)]
    if mpi.is_main:
        print(f"dt: {dt:.3e}, dx: {dx:.3e}, n: {n}, range: {grid_range}")

    # create the problem to solve
    grid = pde.CartesianGrid(grid_range, shape, periodic=[True, False])
    terrain = pde.ScalarField(grid, np.fromfunction(lambda _, y: y * slope, shape))

    bc_zero_derivative = {"derivative": 0}
    bc_zero_flux = {"value": 0}
    bc_periodic = "auto_periodic_neumann"
    bc = [bc_periodic, [bc_zero_flux, bc_zero_derivative]]

    eq = ModelPDE(bc, terrain, p=percipitation)
    solver = ExplicitMPISolver(eq)
    storage = pde.FileStorage(output)

    b = pde.ScalarField.random_uniform(grid, 0, 1e-6)
    w = pde.ScalarField(grid, eq._p / eq._nu)
    h = pde.ScalarField(grid, eq._p / eq._a)
    state = pde.FieldCollection([b, w, h])

    controller = Controller(
        solver, t_range=tmax, tracker=["progress", storage.tracker(1)]
    )
    controller.run(state, dt=dt)
    
    # write_field(output, terrain.data, "terrain")
    
    if mpi.is_main:
        with h5py.File(output, "a") as f:
            f.create_dataset("terrain", data=terrain.data)
            f.create_dataset("dx", data=dx)
            f.create_dataset("dt", data=dt)
            f.create_dataset("p", data=percipitation)