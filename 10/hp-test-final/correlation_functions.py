import numpy as np
from pydpmd.data import load
import os
from numba import njit
from scipy.special import j0
from pydpmd.calc import run_binned, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields

@njit(cache=True, fastmath=True)
def _hist_pairs_pbc(pos, Lx, Ly, edges):
    n = pos.shape[0]
    nb = edges.shape[0] - 1
    counts = np.zeros(nb, dtype=np.int64)
    rmax = edges[-1]
    for i in range(n - 1):
        xi = pos[i, 0]
        yi = pos[i, 1]
        for j in range(i + 1, n):
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            r = (dx * dx + dy * dy) ** 0.5
            if r <= 0.0 or r >= rmax:
                continue
            k = np.searchsorted(edges, r) - 1
            if 0 <= k < nb:
                counts[k] += 1
    return counts

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
def isf_kernel(indices, get_frame, system_id, system_size, system_offset, filters, ks):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    dr = r1 - r0
    dr -= (np.add.reduceat(dr, system_offset[:-1], axis=0) / system_size[:, None])[system_id]  # subtract off the mean drift of each system
    dr = np.linalg.norm(dr, axis=1)
    isf = []
    for f, k in zip(filters, ks):
        # average over all angles (reduces to a bessel function in the infinite limit)
        sq = j0(k * dr[f])
        # average over systems (extract the real part first)
        filtered_system_size = np.bincount(system_id[f])
        isf.append(np.bincount(system_id[f], weights=np.real(sq)) / filtered_system_size)
    return np.array(isf)

@requires_fields("angle")
def angular_isf_kernel(indices, get_frame, system_id, filters, theta_0s):
    t0, t1 = indices
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dtheta = theta1 - theta0
    angle_isf = []
    for f, theta_0 in zip(filters, theta_0s):
        # average over systems (extract the real part first)
        filtered_system_size = np.bincount(system_id[f])
        sq_theta = np.exp(1j * theta_0 * dtheta[f])
        angle_isf.append(np.bincount(system_id[f], weights=np.real(sq_theta)) / filtered_system_size)
    return np.array(angle_isf)

@requires_fields("pos")
def msd_kernel(indices, get_frame, system_id, system_size, system_offset):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    dr = r1 - r0
    dr -= (np.add.reduceat(dr, system_offset[:-1], axis=0) / system_size[:, None])[system_id]  # subtract off the mean drift of each system
    return np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size

@requires_fields("pos", "angle")
def fused_msd_kernel(indices, get_frame, system_id, system_size, system_offset):
    t0, t1 = indices
    r0 = get_frame(t0)['pos']
    r1 = get_frame(t1)['pos']
    theta0 = get_frame(t0)['angle']
    theta1 = get_frame(t1)['angle']
    dr = r1 - r0
    dr -= (np.add.reduceat(dr, system_offset[:-1], axis=0) / system_size[:, None])[system_id]  # subtract off the mean drift of each system
    msd = np.bincount(system_id, weights=np.sum(dr ** 2, axis=-1)) / system_size
    dtheta = (theta1 - theta0)
    angular_msd = np.bincount(system_id, weights=dtheta ** 2) / system_size
    return np.column_stack([msd, angular_msd])

@requires_fields("vel")
def vacf_kernel(indices, get_frame, system_id, system_size):
    t0, t1 = indices
    v0 = get_frame(t0)['vel']
    v1 = get_frame(t1)['vel']
    return np.bincount(system_id, weights=np.sum(v0 * v1, axis=1)) / system_size

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

@requires_fields("pos")
def pair_correlation_function_kernel(indices, get_frame, system_id, box_size, rad, bins):
    t0 = indices[0]
    pos = get_frame(t0)['pos']

    # Expect bin edges for reproducibility/speed; if an int is given, build global edges.
    if np.isscalar(bins):
        # Use global r_max = min(Lx, Ly)/2 across systems so edges are consistent
        rmax = float(np.min(np.min(box_size, axis=1) / 2.0))
        edges = np.linspace(0.0, rmax, int(bins) + 1, dtype=np.float64)
    else:
        edges = np.asarray(bins, dtype=np.float64)
    nb = edges.size - 1
    shell_area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    g_values = []
    for sid in np.unique(system_id):
        Lx, Ly = float(box_size[sid, 0]), float(box_size[sid, 1])
        area = Lx * Ly
        idx_s = (system_id == sid)
        r_s = rad[idx_s]
        p_sys = pos[idx_s]
        g_local = []
        for val in np.unique(r_s):
            f = (r_s == val)
            p = np.ascontiguousarray(p_sys[f], dtype=np.float64)
            n = p.shape[0]
            if n < 2:
                g_local.append(np.zeros(nb, dtype=np.float64))
                continue
            counts = _hist_pairs_pbc(p, Lx, Ly, edges).astype(np.float64)
            counts *= 2.0  # convert i<j counts to ordered-pair counts to match your normalization
            norm = shell_area * (n * (n - 1)) / area
            g_local.append(counts / norm)
        g_values.append(g_local)
    return np.asarray(g_values)

def compute_pair_correlation_function(data, radial_bins=None, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['G'], np.load(save_path)['r']
    if radial_bins is None:
        radial_bins = np.linspace(0.5, 3, 1000)
    bins = TimeBins.from_source(data.trajectory)
    res = run_binned(pair_correlation_function_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'box_size': data.box_size, 'rad': data.rad, 'bins': radial_bins}, show_progress=True, n_workers=20)
    r = (radial_bins[1:] + radial_bins[:-1]) / 2
    G = np.mean(res.mean, axis=0)
    if save_path is not None:
        np.savez(save_path, G=G, r=r)
    return G, r

def compute_vacf(data, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['vacf'], np.load(save_path)['t']
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(vacf_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, vacf=res.mean, t=bins.values())
    return res.mean, bins.values()


def compute_msd(data, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['msd'], np.load(save_path)['t']
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(msd_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size, 'system_offset': data.system_offset}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, msd=res.mean, t=bins.values())
    return res.mean, bins.values()

def compute_rotational_msd(data, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['msd'], np.load(save_path)['t']
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(fused_msd_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size, 'system_offset': data.system_offset}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, msd=res.mean, t=bins.values())
    return res.mean, bins.values()

def compute_shear_modulus(data, save_path=None, subtract_mean_stress=True, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['shear_modulus'], np.load(save_path)['t']
    temp = np.array([data.trajectory[i].temperature for i in range(data.trajectory.num_frames())])
    area = np.prod(data.box_size, axis=-1)
    if subtract_mean_stress:
        mean_stress = np.mean([
            (data.trajectory[i].stress_tensor_total_x[:, 1] + data.trajectory[i].stress_tensor_total_y[:, 0]) / 2.0
            for i in range(data.trajectory.num_frames())
        ], axis=0)
    else:
        mean_stress = 0
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(shear_modulus_kernel, data.trajectory, bins, kernel_kwargs={'stress_xy_mean': mean_stress}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, shear_modulus=res.mean * area / np.mean(temp, axis=0), t=bins.values())
    return res.mean * area / np.mean(temp, axis=0), bins.values()

def compute_isf(data, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['isf'], np.load(save_path)['t']
    filters = [data.rad == r for r in np.unique(data.rad)]
    ks = [2 * np.pi / (r * 2) for r in np.unique(data.rad)]
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(isf_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'system_size': data.system_size, 'system_offset': data.system_offset, 'filters': filters, 'ks': ks}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, isf=res.mean, t=bins.values())
    return res.mean, bins.values()

def compute_angular_isf(data, save_path=None, overwrite=False):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['isf'], np.load(save_path)['t']
    filters = [data.rad == r for r in np.unique(data.rad)]
    theta_0s = [2 * np.pi / (2 * np.pi / n) for n in np.unique(data.n_vertices_per_particle)]
    bins = LagBinsPseudoLog.from_source(data.trajectory)
    res = run_binned(angular_isf_kernel, data.trajectory, bins, kernel_kwargs={'system_id': data.system_id, 'filters': filters, 'theta_0s': theta_0s}, show_progress=True, n_workers=10)
    if save_path is not None:
        np.savez(save_path, isf=res.mean, t=bins.values())
    return res.mean, bins.values()