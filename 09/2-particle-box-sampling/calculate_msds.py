import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.utils import join_systems, split_systems
from pydpmd.plot import draw_particles_frame, create_animation, downsample
import matplotlib.pyplot as plt
import subprocess
import os
from tqdm import tqdm
from pydpmd.calc import run_binned, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import h5py
from matplotlib.colors import LogNorm

def embed_anbles(angle, angular_period, mask=None):
    if mask is None:
        mask = angular_period > 0
    new_angle = np.zeros_like(angle)
    new_angle[mask] = angle[mask] * 2 * np.pi / angular_period[mask]
    new_angle[~mask] = 0
    return np.column_stack([np.cos(new_angle), np.sin(new_angle)])

def compute_generalized_pos(pos, angle, system_id, angular_period, box_size=None, mask=None):
    if box_size is None:
        L = 1
    else:
        L = box_size[system_id]
    if mask is None:
        mask = angular_period > 0
    new_angle = np.zeros_like(angle)
    new_angle[mask] = angle[mask] * 2 * np.pi / angular_period[mask]
    new_angle[~mask] = 0
    return np.column_stack([pos / L, np.cos(new_angle), np.sin(new_angle)])


@requires_fields("pos", "angle")
def masked_fused_msd_kernel(indices, get_frame, system_id, system_size, mask):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dr = r1 - r0
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    dtheta = (theta1 - theta0) * mask
    angular_msd = np.bincount(system_id, weights=dtheta ** 2) / system_size
    return np.column_stack([msd, angular_msd])

@requires_fields("pos", "angle")
def fused_generalized_msd_kernel(indices, get_frame, system_id, system_size, angular_period, box_size):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    gen_pos0 = compute_generalized_pos(r0, theta0, system_id, angular_period, box_size)
    gen_pos1 = compute_generalized_pos(r1, theta1, system_id, angular_period, box_size)
    dr = gen_pos1 - gen_pos0
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    return msd

@requires_fields("pos", "angle")
def fused_msd_generalized_angular_msd_kernel(indices, get_frame, system_id, system_size, angular_period, box_size):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    gen_theta0 = embed_anbles(theta0, angular_period)
    gen_theta1 = embed_anbles(theta1, angular_period)
    dr = r1 - r0
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    dtheta = gen_theta1 - gen_theta0
    angular_msd = np.bincount(system_id, weights=np.sum(dtheta ** 2, axis=-1)) / system_size
    return np.column_stack([msd, angular_msd])

if __name__ == "__main__":
    data_root = "/home/mmccraw/dev/data/09-09-25/bumpy-final"
    mu_eff_paths = [p for p in os.listdir(data_root) if '0.' in p]
    for i, mu_eff in enumerate(mu_eff_paths):
        path = os.path.join(data_root, mu_eff)
        for temperature_path in tqdm(os.listdir(path)):
            if 'dynamics_' not in temperature_path or os.path.isfile(os.path.join(path, temperature_path)):
                continue
            data_path = os.path.join(path, temperature_path)
            try:
                data = load(data_path, location=['init', 'final'], load_trajectory=True, load_full=False)
                msd_path = data_path + '_msd.npz'
                if os.path.exists(msd_path):
                    continue
                bins = LagBinsPseudoLog.from_source(data.trajectory)
                res = run_binned(masked_fused_msd_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size, 'mask': data.moment_inertia > 0}, show_progress=True, n_workers=10)
                _, ids = np.unique(data.delta_phi, return_inverse=True)
                np.savez(msd_path, msd=res.mean, t=bins.values(), ids=ids)
            except Exception as e:
                print(e)
                print(data_path)
                continue