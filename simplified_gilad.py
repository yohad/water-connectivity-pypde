import numpy as np
import numba as nb
import pde
from pde.tools import mpi
from pde.solvers import Controller, ExplicitMPISolver
import time


class ModelPDE(pde.PDEBase):
    def __init__(self, bc, nu=10 / 3, eta=3.5, rho=0.95, gamma=50 / 3, delta_b=1 / 30,
                 delta_w=10 / 3, delta_h=1000 / 3, a=33.33, q=0.05, f=0.1, p=0.5):
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
        p = self._p

        laplace = state.grid.make_operator("laplace", bc=self._bc)

        @nb.jit
        def pde_rhs(state_data, t):
            # state
            b = state_data[0]
            w = state_data[1]

            rate = np.empty_like(state_data)
            # Calculate constants
            L2 = np.float_power(1 + eta * b, 2)
            gb = nu * w * L2
            gw = gamma * b * L2

            # Calculate time derivatives
            rate[0] = gb * b * (1 - b) - b + delta_b * laplace(b)
            rate[1] = p - nu * (1 - rho * b) * w - gw * \
                w + delta_w * laplace(w)

            return rate

        return pde_rhs

    def evolution_rate(self, state: pde.FieldBase, t: float = 0) -> pde.FieldBase:
        b = state[0]
        w = state[1]

        # Calculate constants
        l2 = np.float_power(1 + self._eta * b, 2)
        gb = self._nu * w * l2
        gw = self._gamma * b * l2

        # Calculate time derivatives
        b_t = gb * b * (1 - b) - b + self._delta_b * b.laplace(self._bc)
        w_t = self._p - self._nu * (1 - self._rho * b) * w - gw * w + \
            self._delta_w * w.laplace(self._bc)

        return pde.FieldCollection([b_t, w_t])


def terrain(coords):
    return coords[:, :, 0] / 32


def main():
    percipitation = 1.1
    L = 10
    years = 500
    n = 128
    dx = L / n
    dt = 0.5 * np.power(dx, 2) * 1e-1
    shape = (n, n)
    grid_range = [(0, L), (0, L)]
    if mpi.is_main:
        print(f"dt: {dt}, dx: {dx}, n: {n}, range: {grid_range}")
    grid = pde.CartesianGrid(grid_range, shape, periodic=[True, True])
    b = pde.ScalarField(
        grid, 0.1 + np.fromfunction(lambda x, y: (x + y) / 10000, shape))
    w = pde.ScalarField(grid, 1)
    state = pde.FieldCollection([b, w])

    bc = "auto_periodic_neumann"

    t = time.localtime()
    timestamp = time.strftime("%m%d_%H%M%S", t)
    backup_path = "data\storage-" + timestamp
    storage = pde.FileStorage(backup_path, write_mode="append")

    eq = ModelPDE(p=percipitation, bc=bc)

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
