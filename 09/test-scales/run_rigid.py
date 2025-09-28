import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

if __name__ == "__main__":
    root = "/home/mmccraw/dev/data/09-27-25/test-scales"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    temperature = 1e-4
    n_steps = 1e6
    save_freq = 1e2


    mu_effs = []
    nvs = []
    for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for nv in [3, 6, 10, 20, 30]:
            mu_effs.append(mu_eff)
            nvs.append(nv)
    n_duplicates = len(mu_effs)
    cap_nv = 3
    add_core = True
    rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, n_duplicates)

    dynamics_data_path = os.path.join(root, "rigid", f"T_{temperature:.3e}")
    if not os.path.exists(dynamics_data_path):
        os.makedirs(dynamics_data_path)
    rb.set_velocities(temperature, np.random.randint(0, 1e9))
    # rb.set_neighbor_method(NeighborMethod.Naive)
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(dynamics_data_path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
        dynamics_data_path,
        dynamics_data_path,
        str(n_steps),
        str(save_freq)
    ], check=True)

