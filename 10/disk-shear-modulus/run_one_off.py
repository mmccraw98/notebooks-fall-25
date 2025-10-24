import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_shear_modulus

if __name__ == "__main__":
    for i in range(1):
        root = f"/home/mmccraw/dev/data/10-18-25/example-homework-data-small/"  # run again at 1k particles
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(10, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.5
        temperature = 1e-3
        n_steps = 1e5
        save_freq = 5e1
        dt = 1e-2

        # build the initial data and equilibrate it, ideally to a 0-overlap state

        temp_path = tempfile.mkdtemp()
        # build the initial data and equilibrate it, ideally to a 0-overlap state
        disk = build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9)))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(temp_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_equilibrate_pbc"),
            temp_path,
            temp_path,
        ], check=True)

        disk = load(temp_path, location=["final", "init"])
        shutil.rmtree(temp_path)

        init_path = os.path.join(root, "init")
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(init_path)
        
        # run the iterative compress-dynamics protocol
        # compress the data, incrementing the packing fraction by phi_increment and maintaining the temperature
        disk = load(init_path, location=["final", "init"])
        shutil.rmtree(init_path)
        
        compression_path = os.path.join(root, f"compression")
        disk.set_velocities(temperature, np.random.randint(0, 1e9))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(compression_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nvt_rescale_disk_pbc_compress"),
            compression_path,
            compression_path,
            str(int(max(int(n_steps / 10), 1e3) / 2)),  # total number of steps - half are used for compression, the other half are used for equilibration
            str(0),
            str(temperature),
            str(dt),
        ], check=True)

        # run nve dynamics from the output of the compression data
        disk = load(compression_path, location=["final", "init"])
        dynamics_data_path = os.path.join(root, f"dynamics")
        disk.set_velocities(temperature, np.random.randint(0, 1e9))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(dynamics_data_path)
        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
            dynamics_data_path,
            dynamics_data_path,
            str(n_steps),  # total number of dynamics steps
            str(save_freq),  # save frequency
            str(dt),
        ], check=True)

