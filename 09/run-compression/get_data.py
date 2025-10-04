import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
from correlation_functions import compute_msd, compute_shear_modulus, compute_rotational_msd, compute_pair_correlation_function, compute_vacf, compute_rotational_msd
from pydpmd.calc import run_binned, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
from tqdm import tqdm
import pandas as pd

hp_root = '/home/mmccraw/dev/data/09-27-25/finding-hard-particle-limit/'
avg_dim_overlap_hist = []
nv_hist = []
mu_hist = []
temp_hist = []
p_hist = []
pe_per_particle_hist = []
phi_hist = []
msd_hist = []
t_hist = []
for T_root in os.listdir(hp_root):
    T_root = os.path.join(hp_root, T_root)
    for run_name in os.listdir(T_root):
        if 'dynamics_' not in run_name:
            continue
        run_dir = os.path.join(T_root, run_name)
        try:
            data = load(run_dir, location=['final', 'init'], load_trajectory=True, load_full=True)
        except:
            continue
        data.calculate_mu_eff()
        overlaps = data.trajectory.overlaps.copy()
        summed_overlaps = np.sum(np.add.reduceat(overlaps, data.vertex_system_offset[:-1], axis=1), axis=0)
        avg_overlap = summed_overlaps[:, 0] / summed_overlaps[:, 1]
        vertex_diam = 2 * data.vertex_rad[data.vertex_system_offset[:-1]]
        avg_dimless_overlap = avg_overlap / vertex_diam
        nv = data.n_vertices_per_particle[data.system_offset[:-1]]
        mu = data.mu_eff[data.system_offset[:-1]]
        temp = np.mean(data.trajectory.temperature, axis=0)
        p = np.mean(data.trajectory.pressure, axis=0)
        pe_per_particle = np.mean(data.trajectory.pe_total, axis=0) / data.system_size
        phi = data.packing_fraction.copy()

        msd_path = os.path.join(run_dir, 'msd.npz')
        msd, t = compute_msd(data, msd_path)

        avg_dim_overlap_hist.append(avg_dimless_overlap)
        nv_hist.append(nv)
        mu_hist.append(mu)
        temp_hist.append(temp)
        p_hist.append(p)
        pe_per_particle_hist.append(pe_per_particle)
        phi_hist.append(phi)
        msd_hist.append(msd)
        t_hist.append(t)

avg_dim_overlap_hist = np.array(avg_dim_overlap_hist)
nv_hist = np.array(nv_hist)
mu_hist = np.array(mu_hist)
temp_hist = np.array(temp_hist)
p_hist = np.array(p_hist)
pe_per_particle_hist = np.array(pe_per_particle_hist)
phi_hist = np.array(phi_hist)
msd_hist = np.array(msd_hist)
t_hist = np.array(t_hist)

df = pd.DataFrame({
    'avg_dim_overlap': avg_dim_overlap_hist.flatten(),
    'pe_per_particle': pe_per_particle_hist.flatten(),
    'p': p_hist.flatten(),
    'phi': phi_hist.flatten(),
    'nv': nv_hist.flatten(),
    'mu': mu_hist.flatten(),
    'temp': temp_hist.flatten(),
})
df.to_csv(os.path.join(hp_root, 'aggregated-data.csv'), index=False)