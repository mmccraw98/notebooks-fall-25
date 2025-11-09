import pydpmd as dp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import h5py
from tqdm import tqdm
from pydpmd.plot import draw_particles_frame

def read_data_from_ragged(f, key):
    data_dict = f[f'ragged/{key}']
    sorted_pd_keys = sorted(list(data_dict.keys()), key=lambda s: int(s.rsplit('_', 1)[1]))
    return [data_dict[k][()] for k in sorted_pd_keys]

def get_idx_in_B(A, B):
    mapping = { (x, y): i for i, (x, y) in enumerate(B) }
    return np.fromiter((mapping[tuple(row)] for row in A), dtype=int, count=A.shape[0])

def read_ragged(path):
    with h5py.File(path, 'r') as f:
        pair_dist = read_data_from_ragged(f, 'pair_dist')
        pair_forces = read_data_from_ragged(f, 'pair_forces')
        pair_ids = read_data_from_ragged(f, 'pair_ids')
    unique_pair_ids = np.unique(np.concatenate(pair_ids), axis=0)

    pair_dist_full = np.zeros((len(pair_dist), len(unique_pair_ids)))
    pair_forces_full = np.zeros((len(pair_dist), len(unique_pair_ids), 2))
    pair_ids_full = np.array([unique_pair_ids for i in range(len(pair_dist))])

    for i in tqdm(range(len(pair_dist))):
        mapped_ids = get_idx_in_B(pair_ids[i], pair_ids_full[i])
        pair_forces_full[i, mapped_ids] = pair_forces[i]
        pair_dist_full[i, mapped_ids] = pair_dist[i]
    return pair_dist_full, pair_forces_full, pair_ids_full

root = '/home/mmccraw/dev/data/11-06-25/effective_potential/N-100/'

if __name__ == "__main__":
    for ptype in os.listdir(root):
        path = os.path.join(root, ptype, 'jamming_2')
        pair_dist, pair_force, pair_ids = read_ragged(os.path.join(path, 'trajectory.h5'))
        np.savez(
            os.path.join(path, 'pairs.npz'),
            pair_dist=pair_dist,
            pair_force=pair_force,
            pair_ids=pair_ids
        )