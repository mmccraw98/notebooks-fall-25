import pydpmd as dp
from pydpmd.plot import draw_particles_frame, create_animation, downsample, draw_circle
from pydpmd.utils import split_systems, join_systems
import matplotlib.pyplot as plt
import numpy as np
import os
from mode_resources import *
from system_building_resources import *
import subprocess
import h5py
from collections import defaultdict
from correlation_functions import compute_stress_acf, compute_einstein_helfand_stress_acf
from scipy.interpolate import make_smoothing_spline
import pandas as pd
from matplotlib.colors import LogNorm

temperature_range = np.linspace(1e-5, 2e-5, 20)
delta_phi_range = [-1e-5, -1e-4, -1e-3, -1e-2, -1e-1, -2e-1, -3e-1, -4e-1]
root = '/home/mmccraw/dev/data/11-06-25/'
jamming_root = os.path.join(root, 'jamming', 'N-100')
dynamics_root = os.path.join(root, 'dynamics', 'N-100')

for p in os.listdir(jamming_root):
    if '.DS_Store' in p:
        continue
    first_path_name = os.listdir(os.path.join(jamming_root, p))[0]
    first_path = os.path.join(jamming_root, p, first_path_name)
    dynamics_path = os.path.join(dynamics_root, p, first_path_name)
    if not os.path.exists(dynamics_path):
        os.makedirs(dynamics_path)
    else:
        continue

    data = dp.data.load(first_path, location=['final', 'init'])
    for fname in ['friction_coeff', 'hessian_tt', 'hessian_tx', 'hessian_ty', 'hessian_xt', 'hessian_xx', 'hessian_xy', 'hessian_yt', 'hessian_yx', 'hessian_yy', 'pair_forces', 'pair_ids', 'pair_vertex_contacts']:
        del data.final.arrays[fname]
    
    data_new = []
    temp_list = []
    dphi_list = []
    sid_list = []
    for temperature in temperature_range:
        for delta_phi in delta_phi_range:
            temp_list.extend(np.ones(data.n_systems()) * temperature)
            dphi_list.extend(np.ones(data.n_systems()) * delta_phi)
            sid_list.extend(np.arange(data.n_systems()))
            data_new.append(data)

    data_new = join_systems(data_new)
    data_new.add_array(np.array(temp_list), 'target_temp')
    data_new.add_array(np.array(dphi_list), 'delta_phi')
    data_new.add_array(np.array(sid_list), 'original_system_id')
    data_new.set_velocities(temp_list, np.random.randint(0, 1e9))
    data_new.scale_to_packing_fraction(np.array(dphi_list) + data_new.packing_fraction)

    data_new.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(data_new, 0.3)
    data_new.save(dynamics_path)

    n_steps = 1e5
    save_freq = 1e2
    dt = 1e-2

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
        dynamics_path,
        dynamics_path,
        str(n_steps),
        str(save_freq),
        str(dt),
    ], check=True)
