import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

# create some initial data
# equilibrate the initial data
# define a target packing fraction, a temperature, and a packing fraction increment
# until all systems have a packing fraction greater than the target, run the following:
# compress the systems by the packing fraction increment, maintaining the temperature
# run nve dynamics from the output of the compression data
# repeat

if __name__ == "__main__":
    for temperature, (n_steps, save_freq) in zip(
        [1e-4, 1e-5, 1e-6],
        [(5e4, 5e1), (1e5, 1e2), (5e5, 5e2)]):
        root = f"/home/mmccraw/dev/data/09-27-25/finding-hard-particle-limit-dynamics/T_{temperature:.3e}"
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(50, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.6
        phi_increment = 1e-2
        target_packing_fraction = 0.8
        
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
        rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, n_duplicates)
        init_path = os.path.join(root, "init")
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(init_path)
        
        # run the iterative compress-dynamics protocol
        file_index = 0
        while True:
            # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
            rb = load(init_path, location=["final", "init"])
            compression_path = os.path.join(root, f"compression_{file_index}")
            rb.set_velocities(temperature, np.random.randint(0, 1e9))
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(compression_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_rigid_bumpy_pbc_compress"),
                compression_path,
                compression_path,
                str(max(int(n_steps / 10), 1e4)),  # total number of steps - half are used for compression, the other half are used for equilibration
                str(phi_increment),
                str(temperature),
            ], check=True)

            # run nve dynamics from the output of the compression data
            rb = load(compression_path, location=["final", "init"])
            dynamics_data_path = os.path.join(root, f"dynamics_{file_index}")
            rb.set_velocities(temperature, np.random.randint(0, 1e9))
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(dynamics_data_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
                dynamics_data_path,
                dynamics_data_path,
                str(n_steps),  # total number of dynamics steps
                str(save_freq)  # save frequency
            ], check=True)

            rb = load(dynamics_data_path, location=["final", "init"])
            if np.all(rb.init.packing_fraction > target_packing_fraction):
                break

            # increment the file index and reset the init_path
            file_index += 1
            init_path = dynamics_data_path