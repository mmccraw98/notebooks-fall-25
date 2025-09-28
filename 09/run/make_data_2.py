import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

if __name__ == "__main__":
    data_path = "/home/mmccraw/dev/data/09-09-25/rigid-bumpy/test"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    radii = generate_bidisperse_radii(1000, 0.5, 1.4)
    which = 'small'
    packing_fraction = 0.7
    temperature = 1e-5
    phi_increment = 1e-2
    target_packing_fraction = 0.84
    n_duplicates = 1

    import tempfile
    import shutil
    
    temp_path = tempfile.mkdtemp()
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(time.time())) for _ in range(n_duplicates)])
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