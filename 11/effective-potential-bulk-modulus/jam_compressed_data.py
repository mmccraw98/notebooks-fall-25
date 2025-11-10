import pydpmd as dp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import h5py
from tqdm import tqdm
from pydpmd.plot import draw_particles_frame
from system_building_resources import *

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/11-06-25/dynamics-compression'
    jamming_root = '/home/mmccraw/dev/data/11-06-25/jamming'
    df = {
        'nv': [],
        'mu': [],
        'run_id': [],
        'phi': [],
        'pe_total': [],
        'ke_total': [],
    }
    for fname in os.listdir(root):
        if 'compression' not in fname:
            continue
        path = os.path.join(root, fname)
        try:
            data = dp.data.load(path, location=['final', 'init'])
            data.calculate_mu_eff()
            df['nv'].extend(data.n_vertices_per_particle[data.system_offset[:-1]] - 1)
            df['mu'].extend(data.mu_eff[data.system_offset[:-1]])
            df['run_id'].extend([fname.split('_')[-1] for _  in range(data.n_systems())])
            df['phi'].extend(data.packing_fraction)
            df['pe_total'].extend(data.pe_total)
            df['ke_total'].extend(data.ke_total)
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(df)

    data_list = []
    for nv_mu in np.unique(df[['nv', 'mu']], axis=0):
        _df = df[np.all(df[['nv', 'mu']] == nv_mu, axis=1)]
        run_id = _df[_df.pe_total / _df.ke_total > 1].run_id.astype(int).min()
        data = dp.data.load(os.path.join(root, f'compression_{run_id}'), location=['final', 'init'])
        data.calculate_mu_eff()
        for d in split_systems(data):
            if (d.mu_eff[0] == nv_mu[1]) & (d.n_vertices_per_particle[0] - 1 == nv_mu[0]):
                data_list.append(d)
    data_list = join_systems(data_list)
    data_list.scale_to_packing_fraction(data_list.packing_fraction - 0.05)

    data_list.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(data_list, 0.3)
    data_list.save(jamming_root)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_jam_pbc"),
        jamming_root,
        jamming_root,
    ], check=True)