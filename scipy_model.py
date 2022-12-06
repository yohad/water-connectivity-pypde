import numba as nb
import numpy as np

nu = 10 / 3
eta = 3.5
rho = 0.95
gamma = 50 / 3
delta_b = 1 / 30
delta_w = 10 / 3
delta_h = 1000 / 3
a = 33.33
q = 0.05
f = 0.1
p = 0.5

nb.jit
def model(state_data, t):
    # state
    b = state_data[0]
    w = state_data[1]
    h = state_data[2]

    rate = np.empty_like(state_data)
    # Calculate constants
    I = a * (b + q * f) / (b + q)  # Infiltration Rate
    L2 = np.float_power(1 + eta * b, 2)
    Gb = nu * w * L2
    Gw = gamma * b * L2

    # Calculate time derivatives
    rate[0] = Gb * b * (1 - b) - b + delta_b * laplace(b)
    rate[1] = I * h - nu * (1 - rho * b) * w - \
        Gw * w + delta_w * laplace(w)

    J = -2*delta_h*h*grad(h+zeta)
    rate[2] = p - I * h - div(J)

    return rate
