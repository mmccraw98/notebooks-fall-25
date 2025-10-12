import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import subprocess
import argparse

if __name__ == "__main__":
    root = f"/home/mmccraw/dev/data/10-10-25/friction-contacts-torques/"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(100, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.75
    temperature = 1e-6
    n_steps = 1e5
    save_freq = 1e2
    dt = 1e-2

    # build the initial data and equilibrate it, ideally to a 0-overlap state
    mu_effs = []
    nvs = []
    for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for nv in [3, 6, 10, 20, 30]:
            mu_effs.append(mu_eff)
            nvs.append(nv)

    n_duplicates = len(mu_effs)
    cap_nv = 3
    add_core = True
    rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, 'uniform', n_duplicates)
    rb.calculate_mu_eff()
    print(rb.mu_eff[rb.system_offset[:-1]])

    path = os.path.join(root, "run")
    rb.set_velocities(temperature, np.random.randint(0, 1e9))
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
        path,
        path,
        str(n_steps),  # total number of dynamics steps
        str(save_freq),  # save frequency
        str(dt),
    ], check=True)

    subprocess.run([
        'python',
        'post_proc.py',
        '--path', path
    ])