import argparse

import numpy as np

from water_connectivity.models.flat_terrain import FlatTerrain
from scipy import integrate

from water_connectivity.utils import simulation


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--run_time",
                        action="store",
                        dest="run_time",
                        default=3,
                        help="How long should the simulation run (Unit is 1.2[year])")
    parser.add_argument("-n", "--dimensions",
                        action="store",
                        default=128,
                        help="Size of simulation grid")
    parser.add_argument("-o", "--output_path",
                        action="store",
                        dest="output_path",
                        default=".",
                        help="Where should the simulation results be saved to")

    return parser.parse_args()


def run_simulation(model: FlatTerrain, t_end, save_path):
    """
    Find solutions to the PDE x_t = A(x), where A is `model.get_function()`
    :param model: Simulation to run
    :param t_end: End time for the simulation
    :param save_path: Where to save the results
    """
    print("Starting simulation...")
    sim = simulation.Simulation()
    sim.append(model.biomass())

    initial_state = model._state.flatten()
    rk23 = integrate.RK23(model.get_step_function(), 0, initial_state, t_end)
    integrate.
    # Taking steps in the integration until we reach our end time
    while True:
        try:
            rk23.step()
        except RuntimeError:
            break
        sim.append(model.biomass())
        print(rk23.t)

    print(rk23.t)
    sim.visualize(save_path)
    return rk23


def main():
    args = parse_arguments()

    constant_state = np.ones((128, 128, 3)) * 0.5
    model = FlatTerrain(initial_state=constant_state)
    run_simulation(model, args.run_time, args.output_path)


if __name__ == '__main__':
    main()
