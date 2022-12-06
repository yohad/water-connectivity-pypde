import cupy as cp
import argparse
import time
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--size",
                        action="store",
                        dest="size",
                        default=64)
    parser.add_argument("-t", "--tries",
                        action="store",
                        dest="tries",
                        default=10)
    return parser.parse_args()


def get_kernels(x, y, xp, yp, b, eta):
    A = (cp.power(x - xp, 2) + cp.power(y - yp, 2)) / (2 * cp.power(1 + eta * b, 2))
    return 1 / (2 * cp.pi) * cp.exp(-1 * A)


def double_integration(integrand, dx):
    return cp.trapz(cp.trapz(integrand, dx=dx), dx=dx)


def main():
    args = arg_parser()
    size = int(args.size)

    # Get all points in the discretization of the surface + in integration range
    x = cp.linspace(0, 1, size).reshape((-1, 1, 1, 1))
    y = cp.linspace(0, 1, size).reshape((1, -1, 1, 1))
    xp = cp.linspace(0, 1, size).reshape((1, 1, -1, 1))
    yp = cp.linspace(0, 1, size).reshape((1, 1, 1, -1))
    b = cp.ones((size, size), dtype=cp.float64) * 0.5
    eta = 0.3
    kernels = get_kernels(x, y, xp, yp, b, eta)

    w = cp.ones((size, size), dtype=cp.float64) * 0.4
    integrand = kernels * w
    dx = 1 / size  # intagration is in [0, 1] * [0, 1]
    I = double_integration(integrand, dx)


if __name__ == '__main__':
    start = time.time()
    N = 100
    for _ in tqdm(range(N)):
        main()

    print(f"single run time: {(time.time() - start) / N:.4f} seconds")
