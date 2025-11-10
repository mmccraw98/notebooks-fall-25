import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    root = "/home/mmccraw/dev/data/11-06-25/dynamics-compression"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(100, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    phi_increment = 1e-2
    temperature = 1e-5
    n_steps = 1e5
    save_freq = 1e0
    packing_fraction_target = 0.87
    dt = 1e-2

    mu_effs, nvs = [], []
    for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for nv in [5, 7, 11, 13, 19, 31]:
            nvs.append(nv)
            mu_effs.append(mu_eff)

    assert len(mu_effs) == len(nvs)

    n_duplicates = len(mu_effs)
    cap_nv = 3
    add_core = True
    rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, 'uniform', n_duplicates)
    rb.calculate_mu_eff()

    init_path = os.path.join(root, "init")
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(init_path)

    file_index = 0
    while True:
        compression_path = os.path.join(root, f"compression_{file_index}")
        rb.set_velocities(temperature, np.random.randint(0, 1e9))
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(compression_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
            compression_path,
            compression_path,
            str(1e5),  # total number of steps - half are used for compression, the other half are used for equilibration
            str(phi_increment),
            str(temperature),
            str(dt),
        ], check=True)

        # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
        rb = load(compression_path, location=["final", "init"])
        remaining_systems = [
            _ for i, _ in enumerate(split_systems(rb))
            if rb.init.packing_fraction[i] < packing_fraction_target
        ]
        if len(remaining_systems) == 0:
            break
        rb = join_systems(remaining_systems)

        # increment the file index
        file_index += 1

