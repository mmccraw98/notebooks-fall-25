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
    script_root = "/home/mmccraw/dev/dpmd/build/"
    data_root = "/home/mmccraw/dev/data/12-25-25/example-data/"
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.65
    phi_increment = 1e-2
    pressure_target = 1e-2
    temperature = 1e-4
    n_steps = 1e6
    save_freq = 1e2
    dt = 5e-2
    
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
    init_path = os.path.join(data_root, "init")
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(init_path)

    # run the iterative compress-dynamics protocol
    file_index = 0
    while True:
        # run script with arguments:
        # init_path, file_index, n_steps, phi_increment, temperature, dt, pressure_target

        # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
        rb = load(init_path, location=["final", "init"])
        # extract the data root from the init path (for emulating passing only the init_path and file_index to the script)
        data_root = os.path.dirname(init_path)
        shutil.rmtree(init_path)  # remove the init path
        compression_path = os.path.join(data_root, f"compression_{file_index}")
        rb.set_velocities(temperature, np.random.randint(0, 1e9))
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(compression_path)
        subprocess.run([
            os.path.join(script_root, "nvt_rescale_rigid_bumpy_pbc_compress"),
            compression_path,
            compression_path,
            str(int(max(int(n_steps / 10), 1e5)) / 2),  # total number of steps - half are used for compression, the other half are used for equilibration
            str(phi_increment),
            str(temperature),
            str(dt),
        ], check=True)

        # run nve dynamics from the output of the compression data
        rb = load(compression_path, location=["final", "init"])
        dynamics_data_path = os.path.join(data_root, f"dynamics_{file_index}")
        rb.set_velocities(temperature, np.random.randint(0, 1e9))
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(dynamics_data_path)
        subprocess.run([
            os.path.join(script_root, "nve_rigid_bumpy_pbc_final"),
            dynamics_data_path,
            dynamics_data_path,
            str(n_steps),  # total number of dynamics steps
            str(save_freq),  # save frequency
            str(dt),
        ], check=True)

        # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
        rb = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
        mean_pressure = np.mean([rb.trajectory[i].pressure for i in range(rb.trajectory.num_frames())], axis=0)
        remaining_systems = [
            _ for i, _ in enumerate(split_systems(rb))
            if mean_pressure[i] < pressure_target
        ]
        if len(remaining_systems) == 0:
            break
        rb = join_systems(remaining_systems)
        rb.save(init_path)

        # increment the file index
        file_index += 1