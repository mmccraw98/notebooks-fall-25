import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import tempfile
import shutil

if __name__ == "__main__":
    root = "/home/mmccraw/dev/data/09-27-25/test-scales"
    if not os.path.exists(root):
        os.makedirs(root)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    temperature = 1e-3
    n_steps = 1e6
    save_freq = 1e0

    # build the initial data and equilibrate it, ideally to a 0-overlap state
    n_duplicates = 1
    disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(time.time())) for _ in range(n_duplicates)])
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    temp_path = tempfile.mkdtemp()
    disk.save(temp_path)
    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_equilibrate_pbc"),
        temp_path,
        temp_path,
    ], check=True)

    disk = load(temp_path, location=["final", "init"])
    shutil.rmtree(temp_path)

    dynamics_data_path = os.path.join(root, "disk", f"T_{temperature:.3e}_5")
    if not os.path.exists(dynamics_data_path):
        os.makedirs(dynamics_data_path)
    disk.set_velocities(temperature, np.random.randint(0, 1e9))
    # rb.set_neighbor_method(NeighborMethod.Naive)
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(dynamics_data_path)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_disk_pbc_final"),
        dynamics_data_path,
        dynamics_data_path,
        str(n_steps),
        str(save_freq)
    ], check=True)
