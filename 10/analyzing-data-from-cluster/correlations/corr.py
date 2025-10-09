import pydpmd as dp
from pydpmd.plot import draw_particles_frame
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from pydpmd.data import load
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from pydpmd.calc import run_binned, run_binned_ragged, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields

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

def compute_pair_correlation_function(data, radial_bins=None, save_path=None, overwrite=False, n_workers=20):
    if save_path is not None and os.path.exists(save_path) and not overwrite:
        return np.load(save_path)['G'], np.load(save_path)['r']
    if radial_bins is None:
        radial_bins = np.linspace(0.5, 3, 1000)
    bins = TimeBins.from_source(data.trajectory)
    res = run_binned(
        pair_correlation_function_kernel,
        data.trajectory,
        bins,
        kernel_kwargs={
            'system_id': data.system_id,
            'box_size': data.box_size,
            'rad': data.rad,
            'bins': radial_bins
        },
        show_progress=True,
        n_workers=n_workers
    )
    r = (radial_bins[1:] + radial_bins[:-1]) / 2
    G = np.mean(res.mean, axis=0)
    if save_path is not None:
        np.savez(save_path, G=G, r=r)
    return G, r

def nearest_neighbor_pairs(pos, box_size, rmax, target_rad=None, radii=None):
    tree = cKDTree(np.mod(pos, box_size), boxsize=box_size)
    pairs = np.fromiter(tree.query_pairs(r=rmax), dtype=np.dtype([('i',np.int32),('j',np.int32)]))
    i = pairs['i']
    j = pairs['j']
    if target_rad is not None:
        mask = (radii[i] == target_rad) & (radii[j] == target_rad)
        i = i[mask]
        j = j[mask]
    return i, j

@requires_fields('pos')
def neighbor_list_kernel(indices, get_frame, system_id, box_size, rmax, r_bins=None, target_rad=None, radii=None, sid=None):
    t0 = indices[0]
    pos_all = get_frame(t0)['pos']
    neighbor_list = []
    neighbor_size = []
    bin_ids = []
    if sid is not None:
        sids = [sid]
    else:
        sids = np.unique(system_id)
    for sid in sids:
        pos = pos_all[system_id == sid]
        bs = box_size[sid]
        pairs_i, pairs_j = nearest_neighbor_pairs(pos, bs, rmax, target_rad, radii)
        neighbor_list.append(np.column_stack([pairs_i, pairs_j]))
        neighbor_size.append(len(pairs_i))
        if r_bins is not None:
            dr = pos[pairs_i] - pos[pairs_j]
            dr -= np.round(dr / bs) * bs
            distances = np.linalg.norm(dr, axis=1)
            bin_ids.append(np.digitize(distances, r_bins) + sid * len(r_bins))
    if r_bins is not None:
        return np.concatenate(neighbor_list), np.concatenate([[0], np.cumsum(neighbor_size)]), np.concatenate(bin_ids)
    return np.concatenate(neighbor_list), np.concatenate([[0], np.cumsum(neighbor_size)])

def compute_neighbor_list_for_all_frames(data, rmax, r_bins=None, target_rad=None, sid=None, n_workers=10):
    bins = TimeBins.from_source(data.trajectory)
    res = run_binned_ragged(
        neighbor_list_kernel,
        data.trajectory, bins,
        kernel_kwargs={
            'system_id': data.system_id,
            'box_size': data.box_size,
            'rmax': rmax,
            'r_bins': r_bins,
            'target_rad': target_rad,
            'radii': data.rad,
            'sid': sid
        },
        show_progress=True,
        n_workers=n_workers
    )
    neighbor_list_by_frame = [_[0][0] for _ in res.results]
    neighbor_offset_by_frame = [_[0][1] for _ in res.results]
    if r_bins is not None:
        bin_ids_by_frame = [_[0][2] for _ in res.results]
        return neighbor_list_by_frame, neighbor_offset_by_frame, bin_ids_by_frame
    return neighbor_list_by_frame, neighbor_offset_by_frame



# NEW CORRELATION FUNCTION KERNELS:


# delta theta_i (t + dt) * delta theta_j (t + dt)
# normalized by delta theta ^2 
@requires_fields('angle')
def angle_disp_kernel_1(indices, get_frame, neighbor_list_by_frame, bin_ids_by_frame, n_bins, n_systems):
    t0, t1 = indices
    dtheta = get_frame(t1)['angle'] - get_frame(t0)['angle']
    # average the angular velocity products for neighbors (at the time origin) in each bin
    bin_ids = bin_ids_by_frame[t0]
    neighbor_list = neighbor_list_by_frame[t0]
    # sum products by bin_id
    products = dtheta[neighbor_list[:, 0]] * dtheta[neighbor_list[:, 1]]
    # only get neighbors with equal size radii
    w_w_delta = np.bincount(bin_ids, products, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    delta = np.bincount(bin_ids, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    return w_w_delta, delta

@requires_fields('angle')
def angle_disp_kernel_1_denom(indices, get_frame, system_offset, system_size):
    t0, t1 = indices
    dtheta = get_frame(t1)['angle'] - get_frame(t0)['angle']
    return np.add.reduceat(dtheta ** 2, system_offset[:-1]) / system_size

# delta omega_i (t + dt) * delta omega_j (t + dt)
# normalized by delta omega ^2
@requires_fields('angular_vel')
def angle_disp_kernel_2(indices, get_frame, neighbor_list_by_frame, bin_ids_by_frame, n_bins, n_systems):
    t0, t1 = indices
    omega = get_frame(t1)['angular_vel']
    # average the angular velocity products for neighbors (at the time origin) in each bin
    bin_ids = bin_ids_by_frame[t0]
    neighbor_list = neighbor_list_by_frame[t0]
    # sum products by bin_id
    products = omega[neighbor_list[:, 0]] * omega[neighbor_list[:, 1]]
    # only get neighbors with equal size radii
    w_w_delta = np.bincount(bin_ids, products, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    delta = np.bincount(bin_ids, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    return w_w_delta, delta

@requires_fields('angular_vel')
def angle_disp_kernel_2_denom(indices, get_frame, system_offset, system_size):
    t0, t1 = indices
    omega = get_frame(t1)['angular_vel']
    return np.add.reduceat(omega ** 2, system_offset[:-1]) / system_size

# delta omega_i (t + dt) * delta omega_j (t)
# normalized by delta omega ^2
@requires_fields('angular_vel')
def angle_disp_kernel_3(indices, get_frame, neighbor_list_by_frame, bin_ids_by_frame, n_bins, n_systems):
    t0, t1 = indices
    omega1 = get_frame(t1)['angular_vel']
    omega0 = get_frame(t0)['angular_vel']
    # average the angular velocity products for neighbors (at the time origin) in each bin
    bin_ids = bin_ids_by_frame[t0]
    neighbor_list = neighbor_list_by_frame[t0]
    # sum products by bin_id
    products = omega1[neighbor_list[:, 0]] * omega0[neighbor_list[:, 1]]
    # only get neighbors with equal size radii
    w_w_delta = np.bincount(bin_ids, products, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    delta = np.bincount(bin_ids, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    return w_w_delta, delta

@requires_fields('angular_vel')
def angle_disp_kernel_3_denom(indices, get_frame, system_offset, system_size):
    t0, t1 = indices
    omega = get_frame(t1)['angular_vel']
    return np.add.reduceat(omega ** 2, system_offset[:-1]) / system_size

# delta omega_i (t) * delta omega_j (t)
# normalized by delta omega ^2
@requires_fields('angular_vel')
def angle_disp_kernel_4(indices, get_frame, neighbor_list_by_frame, bin_ids_by_frame, n_bins, n_systems):
    t0 = indices[0]
    omega = get_frame(t0)['angular_vel']
    # average the angular velocity products for neighbors (at the time origin) in each bin
    bin_ids = bin_ids_by_frame[t0]
    neighbor_list = neighbor_list_by_frame[t0]
    # sum products by bin_id
    products = omega[neighbor_list[:, 0]] * omega[neighbor_list[:, 1]]
    # only get neighbors with equal size radii
    w_w_delta = np.bincount(bin_ids, products, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    delta = np.bincount(bin_ids, minlength=n_bins * n_systems).reshape(n_systems, n_bins)
    return w_w_delta, delta

@requires_fields('angular_vel')
def angle_disp_kernel_4_denom(indices, get_frame, system_offset, system_size):
    t0  = indices[0]
    omega = get_frame(t0)['angular_vel']
    return np.add.reduceat(omega ** 2, system_offset[:-1]) / system_size