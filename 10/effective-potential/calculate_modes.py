import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
from pydpmd.calc import run_binned, run_binned_ragged, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
import h5py
from mode_resources import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    args = parser.parse_args()
    root = args.root

    critical_rattler_contact_count = 4
    use_forces_for_rattler_check = True

    i = 0
    while os.path.exists(data_path := os.path.join(root, f'calculation_{i}')):
        print(i, 'of ', len(os.listdir(root)) // 2)
        i += 1
        data = load(data_path, location=['final', 'init'])
        if not os.path.exists(os.path.join(data_path, 'modes.npz')):
            H_list, M_list, val_list, vec_list, non_rattler_id_list = get_dynamical_matrix_modes_for_rigid_bumpy(
                data, critical_rattler_contact_count, use_forces_for_rattler_check
            )
            joined_data = {}
            for j, (H, M, val, vec) in enumerate(zip(H_list, M_list, val_list, vec_list)):
                joined_data.update({f'H_{j}': H})
                joined_data.update({f'M_{j}': M})
                joined_data.update({f'val_{j}': val})
                joined_data.update({f'vec_{j}': vec})
            np.savez(
                os.path.join(data_path, 'modes.npz'),
                joined_data
            )