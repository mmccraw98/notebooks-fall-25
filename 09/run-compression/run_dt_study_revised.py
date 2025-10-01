import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

# running the same dt-study, but using the same initial conditions for all dt's

if __name__ == "__main__":
    n_steps_base = 1e6
    save_freq_base = 1e3
    dt_base = 1e-2

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.73
    temperature = 1e-5
    rng_seed = np.random.randint(0, 1e9)
    
    mu_effs = []
    nvs = []
    for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for nv in [3, 6, 10, 20, 30]:
            mu_effs.append(mu_eff)
            nvs.append(nv)
    n_duplicates = len(mu_effs)
    cap_nv = 3
    add_core = True

    for i in range(3, 7):
        rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, n_duplicates)
        rb.set_velocities(temperature, rng_seed)
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)

        for dt in [1e-2, 5e-2, 1e-1][::-1]:
            n_steps = np.round((n_steps_base * dt_base / dt))
            if n_steps < 1e5:
                n_steps = 1e5
            save_freq = int(save_freq_base * dt_base / dt)
            root = f"/home/mmccraw/dev/data/09-27-25/dt-study-revised-final-final-v{i}/dt_{dt:.3e}"
            if not os.path.exists(root):
                os.makedirs(root)
            dynamics_data_path = os.path.join(root, "dynamics")
            rb.save(dynamics_data_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
                dynamics_data_path,
                dynamics_data_path,
                str(n_steps),  # total number of dynamics steps
                str(save_freq),  # save frequency
                str(dt),
            ], check=True)