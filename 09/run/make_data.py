import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

if __name__ == "__main__":
    data_path = "/home/mmccraw/dev/data/09-09-25/rigid-bumpy/test"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.5
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

    dynamics_data_path = f"/home/mmccraw/dev/data/09-09-25/rigid-bumpy/test_dynamics-2"
    rb.set_velocities(1e-3, 0)
    # rb.set_neighbor_method(NeighborMethod.Naive)
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(dynamics_data_path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
        dynamics_data_path,
        dynamics_data_path,
        str(1e2),
        str(1e0)
    ], check=True)

