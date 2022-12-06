import numpy as np
import numba as nb
import pde
from pde.tools import mpi
from pde.solvers import Controller, ExplicitMPISolver
import time


class ModelPDE(pde.PDEBase):
    def __init__(self, bc, terrain, nu=10 / 3, eta=3.5, rho=0.95, gamma=50/3, delta_b=1/30, delta_w=10 / 3, delta_h=1e-3/3, a=33.33, q=0.05, f=0.1, p=0.5):
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
            rate[1] = i * h - nu * (1 - rho * b) * w - gw * \
                w + delta_w * laplace(w)
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
        w_t = i * h - self._nu * (1 - self._rho * b) * w - gw * w + \
            self._delta_w * w.laplace(self._bc)
        h_t = self._p - i * h - j.divergence(self._bc)

        return pde.FieldCollection([b_t, w_t, h_t])


def main():
    percipitation = 1.15
    L = 10
    years = 1500
    n = 64
    dx = L / n
    dt = 0.5 * np.power(dx, 2) * 1e-1
    shape = (n, n)
    grid_range = [(0, L), (0, L)]
    if mpi.is_main:
        print(f"dt: {dt:.3e}, dx: {dx:.3e}, n: {n}, range: {grid_range}")
    grid = pde.CartesianGrid(grid_range, shape, periodic=[True, False])
    b = pde.ScalarField(grid, 0.5 + np.random.random(shape) / 1e4)
    w = pde.ScalarField(grid, 0.5)
    h = pde.ScalarField(grid, 0.5)
    state = pde.FieldCollection([b, w, h])

    terrain = pde.ScalarField(
        grid, np.fromfunction(lambda _, y: y / 1e3, shape))

    bc_dirichlet_zero = {"value": 0}
    bc_zero_derivative = {"derivative": 0}
    bc_periodic = "auto_periodic_neumann"
    bc = [bc_periodic, [bc_dirichlet_zero, bc_zero_derivative]]
    # bc = bc_periodic

    t = time.localtime()
    timestamp = time.strftime("%m%d_%H%M%S", t)
    backup_path = "storage\storage-" + timestamp + ".h5"
    storage = pde.FileStorage(backup_path)

    eq = ModelPDE(bc, terrain, p=percipitation)

    solver = ExplicitMPISolver(eq)
    controller = Controller(solver, t_range=years, tracker=[
                            "progress", storage.tracker(1)])
    sol = controller.run(state, dt=dt)
    if mpi.is_main:
        video_path = f"results\movie-{timestamp}.mp4"
        pde.movie(storage, filename=video_path,
                  plot_args={"cmap": "YlGn"})


if __name__ == '__main__':
    main()
