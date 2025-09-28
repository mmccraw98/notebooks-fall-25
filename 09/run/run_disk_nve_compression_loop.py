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
    root = "/home/mmccraw/dev/data/09-09-25/run-5/disk"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    temperature = 1e-5
    phi_increment = 1e-2
    target_packing_fraction = 0.84
    n_duplicates = 1
    
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(time.time())) for _ in range(n_duplicates)])
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    init_path = os.path.join(root, "init")
    disk.save(init_path)
    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_equilibrate_pbc"),
        init_path,
        init_path,
    ], check=True)
    
    # run the iterative compress-dynamics protocol
    file_index = 0
    while True:
        # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
        disk = load(init_path, location=["final", "init"])
        compression_path = os.path.join(root, f"compression_{file_index}")
        disk.set_velocities(temperature, int(time.time()))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.calculate_packing_fraction()
        disk.save(compression_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_disk_pbc_compress"),
            compression_path,
            compression_path,
            str(1e5),  # total number of steps - half are used for compression, the other half are used for equilibration
            str(phi_increment),
            str(temperature),
        ], check=True)

        # run nve dynamics from the output of the compression data
        disk = load(compression_path, location=["final", "init"])
        dynamics_data_path = os.path.join(root, f"dynamics_{file_index}")
        disk.set_velocities(temperature, int(time.time()))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(dynamics_data_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
            dynamics_data_path,
            dynamics_data_path,
            str(1e4),  # total number of dynamics steps
            str(1e0)  # save frequency
        ], check=True)

        disk = load(dynamics_data_path, location=["final", "init"])
        if np.all(disk.init.packing_fraction > target_packing_fraction):
            break
        
        # increment the file index and reset the init_path
        file_index += 1
        init_path = dynamics_data_path