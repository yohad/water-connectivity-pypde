import numpy as np
import pde
import h5py
from h5py_utils import write_field, read_field

def post_processing(path_raw):
    """
    Process the raw data to get fields to plot
    """
    with h5py.File(path_raw, "r") as f:
        # pypde stores data with xy and not yx as expected everywhere else
        data = np.array(f["data"]).transpose(0, 1, 3, 2)
        b, w, h = data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :]

        ts = np.array(f["times"])
        terrain = np.array(f["terrain"])
    
    results = {"flux": get_water_flux(h, terrain), "vegetation": b, "times": ts}

    return results

def get_water_flux(h, zeta):
    """
    h has the form ts X ny X nx
    zeta is a 2D array of shape ny X nx of the topology
    """
    derivative = np.gradient(h + zeta, axis=1)
    flux = h * derivative
    flux_aggregate = np.sum(flux, axis=2)
    return flux_aggregate