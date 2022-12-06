from functools import partial

import numpy as np
import cupy as cp
import water_connectivity.utils.math as math_utils
import water_connectivity.models.precipitation as precipitation


class FlatTerrain:
    def __init__(self, initial_state, range=(0, 10, 128), nu=10 / 3, eta=3.5, rho=0.95,
                 gamma=50 / 3,
                 delta_b=1 / 30, delta_w=10 / 3, delta_h=1000 / 3, a=33.33, q=0.05, f=0.1):
        self._shape = initial_state.shape
        self._cp_x, self._cp_y, self._cp_xp, self._cp_yp = cp.ix_(*[cp.linspace(*range)] * 4)
        self._np_x, self._np_y, self._np_xp, self._np_yp = np.ix_(*[np.linspace(*range)] * 4)
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

        self._state = initial_state
        self._precipitation_func = precipitation.const_precipitation(0.5)

    def biomass(self):
        return self._state[:, :, 0]

    def soil_water(self):
        return self._state[:, :, 1]

    def surface_water(self):
        return self._state[:, :, 2]

    def _integration_kernel(self):
        """
        function `g` in the paper
        """
        b = cp.asarray(self.biomass())
        x, y, xp, yp = self._cp_x, self._cp_y, self._cp_xp, self._cp_yp
        kernel = 1 / (2 * cp.pi) * cp.exp(
            -1 * (cp.sqrt(cp.power(x - xp, 2) + cp.power(y - yp, 2))) / (2 * (1 + self._eta * cp.power(b, 2))))

        kernel_np = cp.asnumpy(kernel)
        del kernel
        cp._default_memory_pool.free_all_blocks()

        return kernel_np

    def _appr_kernel(self):
        ...

    def _growth_rate(self, kernels):
        """
        Calculate current growth rate (G_b)
        :return: 2D matrix of growth rate in current time
        """
        integrand = kernels * self.soil_water()
        integration = np.trapz(np.trapz(integrand, dx=10 / 128), dx=10 / 128)
        return integration

    def _soil_water_consumption(self, kernels):
        """
        Calculate current soil water consumption rate (G_w)
        :return: 2D matrix of growth rate in current time
        """
        integrand = kernels * self.biomass()
        integration = np.trapz(np.trapz(integrand, self._np_yp.flatten()), self._np_xp.flatten())
        return integration

    def _update_state(self, u):
        self._state = u.reshape(self._shape)

    def _step(self, t, u):
        # Because RK45 method must take a (n,) vector as `S` we reshape it in here for calculations
        self._update_state(u)

        b = self.biomass()
        w = self.soil_water()
        h = self.surface_water()

        # Calculate constants
        I = self._a * (b + self._q * self._f) / (b + self._q)  # Infiltration Rate
        kernels = self._integration_kernel()
        # Gb = self._growth_rate(kernels)
        # Gw = self._soil_water_consumption(kernels)
        L2 = np.float_power(1 + self._eta * b, 2)
        Gb = self._nu * w * L2
        Gw = self._gamma * b * L2

        # Calculate time derivatives
        b_t = Gb * b * (1 - b) - b + self._delta_b * math_utils.laplace_periodic(b)
        w_t = I * h - self._nu * (1 - self._rho * b) * w - Gw * w + self._delta_w * math_utils.laplace_periodic(w)
        h_t = self._precipitation_func(t) - I * h + self._delta_h * math_utils.laplace_periodic(np.power(h, 2))

        return np.array([b_t, w_t, h_t]).flatten()

    def get_step_function(self):
        """
        We return a function that does not depend on `self` to be used in integration
        :return:
        """
        return partial(self._step)
