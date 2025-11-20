import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    for nv in [5, 7, 11, 13, 19, 31]:
        for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0][::-1]:
            
            root = f"/home/mmccraw/dev/data/11-15-25/jamming/rb/mu-{mu_eff}-nv-{nv}"
            if not os.path.exists(root):
                os.makedirs(root)
            else:
                continue

            radii = generate_bidisperse_radii(100, 0.5, 1.4)
            which = 'small'
            packing_fraction = 0.7

            temp = 1e-5

            n_duplicates = 50

            # build the initial data and equilibrate it, ideally to a 0-overlap state
            cap_nv = 3
            add_core = True
            rb = build_rigid_bumpy_system_from_radii(radii, which, [mu_eff for _ in range(n_duplicates)], [nv for _ in range(n_duplicates)], packing_fraction - 0.1, add_core, cap_nv, 'uniform', n_duplicates)
            rb.calculate_mu_eff()

            jamming_data_path = os.path.join(root, "jamming_0")
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(jamming_data_path)

            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_jam_pbc"),
                jamming_data_path,
                jamming_data_path,
            ], check=True)

            for i in range(1, 10):
                disk = load(jamming_data_path, location=['final', 'init'])
                jamming_data_path = os.path.join(root, f"jamming_{i}")
                disk.set_velocities(temp, np.random.randint(0, 1e9))
                disk.set_neighbor_method(NeighborMethod.Cell)
                set_standard_cell_list_parameters(disk, 0.3)
                disk.save(jamming_data_path)

                subprocess.run([
                    os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
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
                    os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_jam_pbc"),
                    jamming_data_path,
                    jamming_data_path,
                ], check=True)