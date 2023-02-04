import argparse
from time import time
import os

import model
import plotter
from pde.tools import mpi


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--run-time",
        action="store",
        dest="run_time",
        default=500,
        help="How long should the simulation run (Unit is 1.2[year])",
    )
    parser.add_argument(
        "-n",
        "--dimensions",
        action="store",
        default=128,
        help="Size of simulation grid",
    )
    parser.add_argument(
        "-l",
        "--side-length",
        action="store",
        default=20,
        help="Length of the side of the region, in meters",
    )
    parser.add_argument(
        "-p",
        "--percipitation",
        action="store",
        default=1.1,
        type=float,
        help="Amount of rain falling (instability point at 1)",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        action="store",
        dest="output",
        default="output",
        help="Where should the simulation results be saved to",
    )

    return parser.parse_args()


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def main():
    args = parse_arguments()

    unique_name = f"output_{int(time())}"

    output_raw = os.path.join(args.output, "raw")
    safe_makedirs(output_raw)
    output_raw = os.path.join(output_raw, unique_name) + ".h5"

    output_video = os.path.join(args.output, "video")
    safe_makedirs(output_video)
    output_video = os.path.join(output_video, unique_name) + ".mp4"

    model.run_simulation(
        output_raw, args.dimensions, args.run_time, args.side_length, args.percipitation
    )

    if mpi.is_main:
        plotter.plot(output_raw, output_video)


if __name__ == "__main__":
    main()
