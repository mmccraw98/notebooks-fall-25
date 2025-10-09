import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import subprocess

# create some initial data
# equilibrate the initial data
# define a target packing fraction, a temperature, and a packing fraction increment
# until all systems have a packing fraction greater than the target, run the following:
# compress the systems by the packing fraction increment, maintaining the temperature
# run nve dynamics from the output of the compression data
# repeat

if __name__ == "__main__":
    script_root = "/home/mmccraw/dev/dpmd/build/"
    data_root = "/home/mmccraw/dev/data/10-05-25/test/"
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.65
    phi_increment = 1e-2
    pressure_target = 1e-2
    temperature = 1e-6
    n_steps = 10_000
    save_freq = 1000
    dt = 5e-2
    file_index = 0
    
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    temp_path = tempfile.mkdtemp()
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    disk = build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9)))
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(temp_path)
    subprocess.run([
        os.path.join(script_root, "disk_equilibrate_pbc"),
        temp_path,
        temp_path,
    ], check=True)

    disk = load(temp_path, location=["final", "init"])
    shutil.rmtree(temp_path)

    init_path = os.path.join(data_root, "init")
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(init_path)

    # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
    disk = load(init_path, location=["final", "init"])
    # extract the data root from the init path (for emulating passing only the init_path and file_index to the script)
    data_root = os.path.dirname(init_path)
    shutil.rmtree(init_path)  # remove the init path
    compression_path = os.path.join(data_root, f"compression_{file_index}")
    disk.set_velocities(temperature, np.random.randint(0, 1e9))
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(compression_path)
    half_steps = 1e3
    print('Running Compression with Rescaling Thermostat')
    subprocess.run([
        os.path.join(script_root, "nvt_rescale_disk_pbc_compress"),
        compression_path,
        compression_path,
        str(half_steps),  # total number of steps - half are used for compression, the other half are used for equilibration
        str(phi_increment),
        str(temperature),
        str(dt),
    ], check=True)

    # run nve dynamics from the output of the compression data
    disk = load(compression_path, location=["final", "init"])
    dynamics_data_path = os.path.join(data_root, f"dynamics_{file_index}")
    disk.set_velocities(temperature, np.random.randint(0, 1e9))
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(dynamics_data_path)
    print('Running NVE Dynamics')
    subprocess.run([
        os.path.join(script_root, "nve_disk_pbc_final"),
        dynamics_data_path,
        dynamics_data_path,
        str(n_steps),  # total number of dynamics steps
        str(save_freq),  # save frequency
        str(dt),
    ], check=True)

    # load the data, split into separate systems, recombining only those that have a packing fraction less than the target
    disk = load(dynamics_data_path, location=["final", "init"], load_trajectory=True, load_full=False)
    mean_pressure = np.mean([disk.trajectory[i].pressure for i in range(disk.trajectory.num_frames())], axis=0)
    remaining_systems = [
        _ for i, _ in enumerate(split_systems(disk))
        if mean_pressure[i] < pressure_target
    ]
    if len(remaining_systems) == 0:
        exit()  # done
    disk = join_systems(remaining_systems)
    disk.save(init_path)

    # increment the file index
    file_index += 1
