import os
import pydpmd as dp
import numpy as np
from pydpmd.utils import split_systems, join_systems
from system_building_resources import *
import subprocess

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/11-01-25/'

    # delta_phis = [-1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1, -2e-1, -3e-1, -4e-1, -5e-1]
    delta_phis = -np.logspace(-2, np.log10(0.2), 10)
    temperature = 1e-6

    for p in os.listdir(os.path.join(root, 'jamming')):
        if 'disk' in p:
            continue
        all_temps, all_delta_phis = [], []
        all_data = []
        data = dp.data.load(os.path.join(root, 'jamming', p, 'jamming'), location=['final', 'init'])
        for fname in ['friction_coeff', 'hessian_tt', 'hessian_tx', 'hessian_ty', 'hessian_xt', 'hessian_xx', 'hessian_xy', 'hessian_yt', 'hessian_yx', 'hessian_yy', 'pair_forces', 'pair_ids', 'pair_vertex_contacts']:
            del data.final.arrays[fname]
        for delta_phi in delta_phis:
            all_temps.append(temperature)
            all_delta_phis.append(delta_phi)
            all_data.append(data)
        data_new = join_systems(all_data)
        print('scaling')
        data_new.scale_to_packing_fraction(data_new.packing_fraction + np.array(all_delta_phis))
        print('setting temp')
        data_new.set_velocities(np.array(all_temps), np.random.randint(0, 1e9))
        data_new.add_array(np.array(all_delta_phis), 'delta_phi')
        data_new.add_array(np.array(all_temps), 'target_temp')

        rb_dynamics_path = os.path.join(root, 'rb-dynamics-pressure', p)
        data_new.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(data_new, 0.3)
        data_new.save(rb_dynamics_path)

        n_steps = 1e5
        save_freq = 1e2
        dt = 1e-2

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
            rb_dynamics_path,
            rb_dynamics_path,
            str(n_steps),
            str(save_freq),
            str(dt),
        ], check=True)