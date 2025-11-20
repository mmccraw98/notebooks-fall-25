import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    root = "/home/mmccraw/dev/data/11-15-25/jamming/disk/"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(100, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    packing_fraction_increment = 1e-2
    temp = 1e-5

    n_duplicates = 50

    # build the initial data and equilibrate it, ideally to a 0-overlap state
    temp_path = tempfile.mkdtemp()
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9))) for _ in range(n_duplicates)])
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

    jamming_data_path = os.path.join(root, "jamming_0")
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(jamming_data_path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_jam_pbc"),
        jamming_data_path,
        jamming_data_path,
    ], check=True)

    for i in range(1, 10):
        disk = load(jamming_data_path, location=['final', 'init'])
        disk.scale_to_packing_fraction(disk.packing_fraction - packing_fraction_increment)
        jamming_data_path = os.path.join(root, f"jamming_{i}")
        disk.set_velocities(temp, np.random.randint(0, 1e9))
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(jamming_data_path)

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
            jamming_data_path,
            jamming_data_path,
            str(1e5),
            str(1e2),
            str(1e-2)
        ], check=True)

        disk = load(jamming_data_path, location=['final', 'init'])
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        shutil.rmtree(jamming_data_path)
        disk.save(jamming_data_path)

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_jam_pbc"),
            jamming_data_path,
            jamming_data_path,
        ], check=True)