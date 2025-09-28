import numpy as np
from pydpmd.data import load
import os
from pydpmd.calc import run_binned, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields

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

@requires_fields("pos")
def msd_kernel(indices, get_frame, system_id, system_size):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    dr = r1 - r0
    return np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size

@requires_fields("pos", "angle")
def fused_msd_kernel(indices, get_frame, system_id, system_size):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dr = r1 - r0
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    dtheta = (theta1 - theta0)
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

@requires_fields("stress_tensor_total_x", "stress_tensor_total_y")
def shear_modulus_kernel(indices, get_frame, stress_xy_mean=0):
    t0, t1 = indices
    s_x_0 = get_frame(t0)['stress_tensor_total_x']
    s_y_0 = get_frame(t0)['stress_tensor_total_y']
    s_x_1 = get_frame(t1)['stress_tensor_total_x']
    s_y_1 = get_frame(t1)['stress_tensor_total_y']
    # autocorrelation of the off-diagonal element of the stress tensor (use the average of the two for hopefully cleaner result)
    off_diag_0 = (s_x_0[:, 1] + s_y_0[:, 0]) / 2 - stress_xy_mean
    off_diag_1 = (s_x_1[:, 1] + s_y_1[:, 0]) / 2 - stress_xy_mean
    return off_diag_0 * off_diag_1

def compute_msd(data, save_path=None):
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(msd_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, msd=res.mean, t=bins.values())
    return res.mean, bins.values()

def compute_rotational_msd(data, save_path=None):
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(fused_msd_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, msd=res.mean, t=bins.values())
    return res.mean, bins.values()

def compute_shear_modulus(data, save_path=None, mean_stress=0):
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(shear_modulus_kernel, data.trajectory, bins, kernel_kwargs={'stress_xy_mean': mean_stress}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, shear_modulus=res.mean, t=bins.values())
    return res.mean, bins.values()
