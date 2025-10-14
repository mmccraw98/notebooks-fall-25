import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    root = "/home/mmccraw/dev/data/10-15-25/animations-large/"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(10000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.75
    temperature = 1e-6
    n_steps = 1e6
    save_freq = 1e3
    dt = 5e-2

    # build the initial data and equilibrate it, ideally to a 0-overlap state
    mu_effs = [1.0]
    nvs = [7]
    assert len(mu_effs) == len(nvs)

    cap_nv = 3
    add_core = True
    # create the systems at a 10% lower density than initially desired so that the duplicate systems have a chance to become sufficiently dissimilar
    # this way, we avoid having to create 10x duplicates in the below function which is very slow
    rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, 'uniform', len(mu_effs))
    rb.calculate_mu_eff()

    init_path = os.path.join(root, "init")
    rb.set_velocities(temperature, np.random.randint(0, 1e9))
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(init_path)

    # load the data and minimize it to the nearest energy minimum
    rb = load(init_path, location=["final", "init"])
    minimization_data_path = os.path.join(root, "minimization")
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(minimization_data_path)
    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_equilibrate_pbc"),
        minimization_data_path,
        minimization_data_path,
    ], check=True)

    rb = load(minimization_data_path, location=["final", "init"])
    run_path = os.path.join(root, "run")
    rb.set_velocities(temperature, np.random.randint(0, 1e9))
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(run_path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final_copy"),
        run_path,
        run_path,
        str(n_steps),  # total number of dynamics steps
        str(save_freq),  # save frequency
        str(dt),
    ], check=True)