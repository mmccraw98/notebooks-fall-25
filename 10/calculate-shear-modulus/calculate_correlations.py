import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
from correlation_functions import compute_msd, compute_shear_modulus, compute_rotational_msd, compute_pair_correlation_function, compute_vacf, compute_rotational_msd
from pydpmd.calc import run_binned, run_binned_ragged, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
from correlation_functions import compute_neighbor_list_for_all_frames, simple_angle_disp_kernel, angle_disp_kernel, angle_disp_kernel_denominator

if __name__ == "__main__":
    rmax = 3
    n_bins = 100
    r_bins = np.linspace(0.5, rmax, n_bins)
    root = '/home/mmccraw/dev/data/10-01-25/short-test-2/trial-0/'

    for path in os.listdir(root):
        if not path.startswith('dynamics_'):
            continue
        data = load(os.path.join(root, path), location=['init', 'final'], load_trajectory=True, load_full=False)
        neighbor_list_by_frame, neighbor_offset_by_frame, bin_ids_by_frame = compute_neighbor_list_for_all_frames(data, rmax, r_bins)
        n_systems = data.n_systems()

        bins = LagBinsPseudoLog.from_source(data.trajectory)

        res_simple = run_binned(simple_angle_disp_kernel, data.trajectory, bins, kernel_kwargs={'system_size': data.system_size, 'system_offset': data.system_offset, 'neighbor_list_by_frame': neighbor_list_by_frame, 'neighbor_offset_by_frame': neighbor_offset_by_frame}, show_progress=True, n_workers=10)
        np.savez(
            os.path.join(root, path, 'simple_angle_disp.npz'),
            simple_angle_disp=res_simple.mean,
            t=bins.values()
        )

        res = run_binned(angle_disp_kernel, data.trajectory, bins, kernel_kwargs={'system_size': data.system_size, 'system_offset': data.system_offset, 'neighbor_list_by_frame': neighbor_list_by_frame, 'bin_ids_by_frame': bin_ids_by_frame, 'n_bins': n_bins, 'n_systems': n_systems}, show_progress=True, n_workers=10)
        res_denominator = run_binned(angle_disp_kernel_denominator, data.trajectory, bins, kernel_kwargs={'system_size': data.system_size, 'system_offset': data.system_offset}, show_progress=True, n_workers=10)
        np.savez(
            os.path.join(root, path, 'angle_disp.npz'),
            angle_disp=res.mean,
            denominator=res_denominator.mean,
            t=bins.values(),
            r=r_bins
        )