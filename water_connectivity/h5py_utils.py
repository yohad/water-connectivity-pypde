import h5py
import numpy as np

def write_field(path, field, name):
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=field)

def read_field(path, name):
    with h5py.File(path, "r") as f:
        return np.array(f[name])