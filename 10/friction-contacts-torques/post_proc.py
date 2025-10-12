import numpy as np
import pickle
import os
from pydpmd.data import load
import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    path = args.path

    data = load(path, location=["final", "init"], load_trajectory=True, load_full=False)
    data.calculate_mu_eff()

    with h5py.File(os.path.join(path, 'trajectory.h5'), 'r') as f:
        friction_coeff = [f['ragged']['friction_coeff'][k][:] for k in sorted(list(f['ragged']['friction_coeff'].keys()))]
        pair_ids = [f['ragged']['pair_ids'][k][:] for k in sorted(list(f['ragged']['pair_ids'].keys()))]
        pair_vertex_contacts = [f['ragged']['pair_vertex_contacts'][k][:] for k in sorted(list(f['ragged']['pair_vertex_contacts'].keys()))]

    # save this:
    n_contacts_total = np.mean([data.trajectory[i].n_contacts_total for i in range(data.trajectory.num_frames())], axis=0)

    abs_torque_hist = np.abs(np.array([data.trajectory[i].torque for i in range(data.trajectory.num_frames())]))
    abs_torque_sum = np.add.reduceat(abs_torque_hist, data.system_offset[:-1], axis=1)
    num_nonzero_torque = np.add.reduceat(abs_torque_hist != 0, data.system_offset[:-1], axis=1)

    # save these:
    abs_torque_mean_nonzero = np.mean(abs_torque_sum / num_nonzero_torque, axis=0)
    abs_torque_std_nonzero = np.std(abs_torque_sum / num_nonzero_torque, axis=0)
    abs_torque_mean = np.mean(abs_torque_sum / data.system_size, axis=0)
    abs_torque_std = np.std(abs_torque_sum / data.system_size, axis=0)

    force_norm_hist = np.linalg.norm(np.array([data.trajectory[i].force for i in range(data.trajectory.num_frames())]), axis=2)
    force_norm_sum = np.add.reduceat(force_norm_hist, data.system_offset[:-1], axis=1)
    num_nonzero_force = np.add.reduceat(force_norm_hist != 0, data.system_offset[:-1], axis=1)

    # save these:
    force_norm_mean_nonzero = np.mean(force_norm_sum / num_nonzero_force, axis=0)
    force_norm_std_nonzero = np.std(force_norm_sum / num_nonzero_force, axis=0)
    force_norm_mean = np.mean(force_norm_sum / data.system_size, axis=0)
    force_norm_std = np.std(force_norm_sum / data.system_size, axis=0)

    friction_coeff_hist = [[] for _ in range(data.n_systems())]
    zv_hist = [[] for _ in range(data.n_systems())]

    for fid in range(len(friction_coeff)):
        friction_coeff_frame = friction_coeff[fid]
        pair_vertex_contacts_frame = pair_vertex_contacts[fid]
        pair_ids_frame = pair_ids[fid]
        for sid in range(data.n_systems()):
            beg = data.system_offset[sid]
            end = data.system_offset[sid + 1]
            system_mask = np.all((pair_ids_frame >= beg) & (pair_ids_frame < end), axis=1)
            friction_coeff_hist[sid].extend(friction_coeff_frame[system_mask])
            zv_hist[sid].append(pair_vertex_contacts_frame[system_mask])
    friction_coeff_hist = [np.array(f) for f in friction_coeff_hist]
    zv_hist = [np.concatenate(f) for f in zv_hist]

    pair_type_count_hist = []
    pair_type_friction_hist = []

    for z, mu in zip(zv_hist, friction_coeff_hist):
        total_counts = z[np.all(z != 0, axis=1)].shape[0]
        pair_types = np.sort(np.unique(z[np.all(z != 0, axis=1)], axis=0), axis=1)
        unique_pairs, inverse_ids = np.unique(pair_types, axis=0, return_inverse=True)
        pair_to_id_map = {tuple(pair): idx for idx, pair in enumerate(unique_pairs)}

        count_map = {}
        friction_map = {}
        for pair, idx in pair_to_id_map.items():
            count_map[idx] = np.all(z == pair, axis=1).sum()
            friction_map[idx] = np.mean(mu[np.all(z == pair, axis=1)])

        pair_type_count_hist.append({i_j: count_map[idx] / total_counts for i_j, idx in pair_to_id_map.items()})
        pair_type_friction_hist.append({i_j: friction_map[idx] for i_j, idx in pair_to_id_map.items()})

    with open(os.path.join(path, 'friction_contacts_torques.pkl'), 'wb') as f:
        pickle.dump({
            'friction_coeff_hist': [mu[mu > 0] for mu in friction_coeff_hist],
            'pair_type_count_hist': pair_type_count_hist,
            'pair_type_friction_hist': pair_type_friction_hist,
            'force_norm_nonzero': force_norm_mean_nonzero,
            'force_norm_std_nonzero': force_norm_std_nonzero,
            'force_norm': force_norm_mean,
            'force_norm_std': force_norm_std,
            'abs_torque_nonzero': abs_torque_mean_nonzero,
            'abs_torque_std_nonzero': abs_torque_std_nonzero,
            'abs_torque': abs_torque_mean,
        }, f)
