import pydpmd as dp
from pydpmd.utils import join_systems
from system_building_resources import *
import numpy as np
import os
import subprocess
import shutil

script_root = '/home/mmccraw/dev/dpmd/build'

size = 'small'
T_0 = 1e-5
T_f = 2e-5
n_T = 50

delta_phi_min = 1e-6
delta_phi_max = 0.2
n_phi_steps = 50

n_steps = int(1e4)
dt = 1e-2
save_freq = 1e2



if __name__ == "__main__":
    source_root = f'/home/mmccraw/dev/data/12-01-25/grace-data/rb-{size}/'
    target_root = f'/home/mmccraw/dev/data/12-01-25/specific-heat/ic-jammed/rb-{size}/T-{T_0:.2e}/'

    for mu_root in os.listdir(source_root):
        for offset in np.logspace(np.log10(delta_phi_min), np.log10(delta_phi_max), n_phi_steps):
            offset_root = f'delta_phi-{offset:.3e}'

            path = os.path.join(source_root, mu_root, 'jamming_9')
            run_path = os.path.join(target_root, mu_root, offset_root)
            if not os.path.exists(run_path):
                os.makedirs(run_path)

            data = dp.data.load(path, location=['init'])
            temps = np.concatenate([np.ones(data.n_systems()) * T for T in np.linspace(T_0, T_f, n_T)])
            ids = np.concatenate([np.arange(data.n_systems()) for T in np.linspace(T_0, T_f, n_T)])
            data = join_systems([data for _ in range(n_T)])
            data.set_velocities(temps, np.random.randint(0, 1e9))
            data.add_array(temps, 'target_temp')
            data.add_array(ids, 'original_id')
            data.add_array(np.array([offset for _ in range(data.n_systems())]), 'delta_phi')
            data.scale_to_packing_fraction(data.packing_fraction - offset)

            # if size == 'small':
                # data.set_neighbor_method(NeighborMethod.Naive)
            # else:
            data.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(data, 0.3)

            data.save(run_path)

            subprocess.run([
                os.path.join(script_root, "nve_rigid_bumpy_pbc_final"),
                run_path,
                run_path,
                str(n_steps),
                str(save_freq),
                str(dt)
            ], check=True)

            break
        break