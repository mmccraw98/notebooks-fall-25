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

    # no-relax: run immediately from the initial jammed config
    # relax: relax in subsequent steps from initial jammed config, then run protocol

    nv = 31
    N = 100
    temperature = 1e-6
    temperatures = np.linspace(temperature, 2 * temperature, 20)
    n_steps = 1e5
    delta_phi_values = -np.logspace(-6, np.log10(0.2), 50)
    for i in range(3):
        for mu in [0.01, 0.05, 0.1, 0.5, 1.0]:
            jamming_root = f'/home/mmccraw/dev/data/11-06-25/jamming/N-{N}/{nv}-{mu:.2f}/jamming_{i}'
            for j, delta_phi in enumerate(delta_phi_values):
                specific_heat_root = f'/home/mmccraw/dev/data/12-25-25/specific_heat_rb/no-relax/N-{N}/{nv}-{mu:.2f}/cv_{i}/'
                if not os.path.exists(specific_heat_root):
                    os.makedirs(specific_heat_root)
                
                # data = []
                # for T in temperatures:
                #     d = dp.data.load(jamming_root, location=['final', 'init'])
                #     d.calculate_mu_eff()
                #     d.set_velocities(T, np.random.randint(0, 1e9))
                #     data.append(d)

                data = dp.data.load(jamming_root, location=['final', 'init'])
                data.calculate_mu_eff()
                join_systems(split_systems(data))
                # data = join_systems(data)

                phi_0 = data.packing_fraction.copy()
                data.add_array(np.ones(data.n_systems()) * delta_phi, 'delta_phi')
                data.set_neighbor_method(NeighborMethod.Cell)
                set_standard_cell_list_parameters(data, 0.3)
                exit()



    exit()



    #     run_root = os.path.join(pressure_root, f'run_{file_id}')
    #     if not os.path.exists(run_root):
    #         data.save(run_root)
            
    #         subprocess.run([
    #             os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
    #             run_root,
    #             run_root,
    #             str(5e4),
    #             str(-np.mean(data.packing_fraction - (phi_0 - _delta_phi))),
    #             str(temperature),
    #             str(1e-2),
    #         ], check=True)

    #         data = dp.data.load(run_root, location=['final', 'init'])
    #         shutil.rmtree(run_root)
    #         data.add_array(np.ones(data.n_systems()) * _delta_phi, 'delta_phi')
    #         data.set_neighbor_method(NeighborMethod.Cell)
    #         set_standard_cell_list_parameters(data, 0.3)
    #         data.save(run_root)

    #         subprocess.run([
    #             os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
    #             run_root,
    #             run_root,
    #             str(n_steps),
    #             str(1e2),
    #             str(1e-2)
    #         ], check=True)

    #     data = dp.data.load(run_root, location=['final', 'init'])

    # pressure_root = '/home/mmccraw/dev/data/11-06-25/pressure-loop/log'
    # cv_root = '/home/mmccraw/dev/data/11-06-25/cv-loop'

    # temperatures = np.linspace(1e-5, 2e-5, 20)
    # n_steps = 1e5

    # for fname in os.listdir(pressure_root):
    #     try:
    #         data = dp.data.load(os.path.join(pressure_root, fname), location=['final', 'init'])
    #     except:
    #         continue
    #     temps = np.concatenate([np.ones(data.n_systems()) * temp for temp in temperatures])
    #     data = join_systems([data for _ in range(len(temperatures))])
    #     data.set_velocities(temps, np.random.randint(0, 1e9))
    #     data.add_array(temps, 'target_temp')

    #     data.set_neighbor_method(NeighborMethod.Cell)
    #     set_standard_cell_list_parameters(data, 0.3)
        
    #     run_root = os.path.join(cv_root, fname)
    #     data.save(run_root)
        
    #     subprocess.run([
    #         os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
    #         run_root,
    #         run_root,
    #         str(n_steps),
    #         str(1e2),
    #         str(1e-2)
    #     ], check=True)