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
    jamming_root = '/home/mmccraw/dev/data/11-06-25/jamming'
    data = dp.data.load(jamming_root, location=['final', 'init'])
    phi_0 = data.packing_fraction.copy()

    temperature = 1e-5
    n_steps = 1e6

    delta_phi = np.logspace(-3, np.log10(0.2), 50)
    pressure_root = '/home/mmccraw/dev/data/11-06-25/pressure-loop/log'

    os.makedirs(pressure_root, exist_ok=True)
    data.set_velocities(temperature, np.random.randint(0, 1e9))
    for file_id, _delta_phi in enumerate(delta_phi):
        data.add_array(np.ones(data.n_systems()) * _delta_phi, 'delta_phi')
        data.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(data, 0.3)
        
        run_root = os.path.join(pressure_root, f'run_{file_id}')
        if not os.path.exists(run_root):
            data.save(run_root)
            
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                run_root,
                run_root,
                str(5e4),
                str(-np.mean(data.packing_fraction - (phi_0 - _delta_phi))),
                str(temperature),
                str(1e-2),
            ], check=True)

            data = dp.data.load(run_root, location=['final', 'init'])
            shutil.rmtree(run_root)
            data.add_array(np.ones(data.n_systems()) * _delta_phi, 'delta_phi')
            data.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(data, 0.3)
            data.save(run_root)

            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
                run_root,
                run_root,
                str(n_steps),
                str(1e2),
                str(1e-2)
            ], check=True)

        data = dp.data.load(run_root, location=['final', 'init'])