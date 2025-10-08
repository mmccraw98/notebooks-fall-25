from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
from correlation_functions import compute_neighbor_list_for_all_frames, simple_angle_disp_kernel, angle_disp_kernel, angle_disp_kernel_denominator
import argparse
import os
import numpy as np
from pydpmd.data import load
from pydpmd.calc import run_binned, LagBinsPseudoLog

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    root = args.root
    path = args.path


    rmax = 3
    n_bins = 100
    r_bins = np.linspace(0.5, rmax, n_bins)

    data = load(os.path.join(root, path), location=['init', 'final'], load_trajectory=True, load_full=False)
    data.calculate_mu_eff()
    n_systems = data.n_systems()

    dr = r_bins[1] - r_bins[0]
    shell_area = np.pi * ((r_bins + dr) ** 2 - r_bins ** 2)
    area = np.prod(data.box_size, axis=1)

    unique_radii = np.unique(data.rad)
    radii = data.rad

    bins = LagBinsPseudoLog.from_source(data.trajectory)

    for target_rad in unique_radii:
        # compute the specific radii neighbor list
        neighbor_list_by_frame, neighbor_offset_by_frame, bin_ids_by_frame = compute_neighbor_list_for_all_frames(data, rmax, r_bins, target_rad=target_rad)
        
        # radial distribution normalization factor
        n = np.add.reduceat(radii == target_rad, data.system_offset[:-1])
        normalization = (shell_area[None, :] * (n * (n - 1) / (2 * area))[:, None])

        res = run_binned(angle_disp_kernel, data.trajectory, bins, kernel_kwargs={'neighbor_list_by_frame': neighbor_list_by_frame, 'bin_ids_by_frame': bin_ids_by_frame, 'n_bins': n_bins, 'n_systems': n_systems, 'normalization': normalization}, show_progress=True, n_workers=10)
        res_denominator = run_binned(angle_disp_kernel_denominator, data.trajectory, bins, kernel_kwargs={'system_size': data.system_size, 'system_offset': data.system_offset}, show_progress=True, n_workers=10)