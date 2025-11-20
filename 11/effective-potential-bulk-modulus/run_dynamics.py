import pydpmd as dp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import h5py
from tqdm import tqdm
from pydpmd.plot import draw_particles_frame
from system_building_resources import *

if __name__ == "__main__":
    pressure_root = '/home/mmccraw/dev/data/11-06-25/pressure-loop/log'
    dynamics_root = '/home/mmccraw/dev/data/11-06-25/dynamics-loop'

    n_steps = 1e6

    for fname in os.listdir(pressure_root):
        data = dp.data.load(os.path.join(pressure_root, fname), location=['final', 'init'])

        data.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(data, 0.3)
        
        run_root = os.path.join(dynamics_root, fname)
        data.save(run_root)
        
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
            run_root,
            run_root,
            str(n_steps),
            str(1e2),
            str(5e-2)
        ], check=True)