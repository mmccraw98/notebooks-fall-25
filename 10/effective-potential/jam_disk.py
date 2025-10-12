import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    
    i = 0
    for name, n_particles, n_duplicates in zip(['small', 'large'], [10, 100], [1000, 100]):
        root = f"/home/mmccraw/dev/data/10-14-25/effective-potential/jamming-{name}/disk/"
        if not os.path.exists(root):
            os.makedirs(root)

        radii = generate_bidisperse_radii(n_particles, 0.5, 1.4)
        which = 'small'
        packing_fraction = 0.6


        # build the initial data and equilibrate it, ideally to a 0-overlap state
        temp_path = tempfile.mkdtemp()
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

        jamming_data_path = os.path.join(root, f"jamming_{i}")
        disk.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(disk, 0.3)
        disk.save(jamming_data_path)

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_jam_pbc"),
            jamming_data_path,
            jamming_data_path,
        ], check=True)
