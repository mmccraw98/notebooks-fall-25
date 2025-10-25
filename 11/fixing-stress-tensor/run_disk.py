import pydpmd as dp
from pydpmd.data import NeighborMethod
import numpy as np
import os
import time
import shutil
from system_building_resources import *
from correlation_functions import compute_stress_acf

if __name__ == "__main__":
    source_path = '/home/mmccraw/dev/data/10-06-25/jamming/disk/jamming_0'

    n_steps = 1e5
    save_freq = 1e0
    dt = 1e-2

    # for delta_phi in [-1e-4, -1e-3, -1e-2]:
        # for temperature in [1e-3, 1e-4, 1e-5]:

    delta_phi = -1e-2
    temperature = 1e-5
    # run nve dynamics from the output of the compression data
    disk = dp.data.load(source_path, location=['final', 'init'])
    target_path = os.path.join(os.path.dirname(source_path), 'dynamics-one-off-3', f'delta_phi_{delta_phi:.3e}_temp_{temperature:.3e}')
    disk.scale_to_packing_fraction(disk.packing_fraction + delta_phi)
    disk.set_velocities(temperature, np.random.randint(0, 1e9))
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(target_path)
    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
        target_path,
        target_path,
        str(n_steps),  # total number of dynamics steps
        str(save_freq),  # save frequency
        str(dt),
    ], check=True)

    disk = dp.data.load(target_path, location=['final', 'init'], load_trajectory=True, load_full=False)
    G, T, A, stress, t = compute_stress_acf(disk, subtract_mean_stress=True)
    G_no_sub, T, A, stress, t = compute_stress_acf(disk, subtract_mean_stress=False)
    shutil.rmtree(target_path)
    np.savez(
        target_path.rstrip('/') + '.npz',
        G=G,
        G_no_sub=G_no_sub,
        T=T,
        A=A,
        stress=stress,
        t=t,
        delta_phi=delta_phi,
        temperature=temperature
    )

