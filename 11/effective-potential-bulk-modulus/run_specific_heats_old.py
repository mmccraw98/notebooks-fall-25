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

    delta_phi, temperature = [], []
    for dphi in np.logspace(-3, np.log10(0.2), 20):
        for temp in np.linspace(1e-5, 2e-5, 20):
            delta_phi.append(dphi)
            temperature.append(temp)
    delta_phi = np.array(delta_phi)
    temperature = np.array(temperature)

    n_steps = 1e5
    
    cv_root = '/home/mmccraw/dev/data/11-06-25/cv-loop'

    os.makedirs(cv_root, exist_ok=True)
    
    for i, d in enumerate(split_systems(data)):
        d = join_systems([d for i in range(len(temperature))])
        d.set_velocities(temperature, np.random.randint(0, 1e9))
        d.scale_to_packing_fraction(d.packing_fraction - delta_phi)
        
        data.add_array(delta_phi, 'delta_phi')
        data.add_array(temperature, 'target_temp')
        
        data.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(data, 0.3)
        
        run_root = os.path.join(cv_root, f'system_{i}')
        data.save(run_root)
        
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
            run_root,
            run_root,
            str(n_steps),
            str(1e2),
            str(1e-2)
        ], check=True)