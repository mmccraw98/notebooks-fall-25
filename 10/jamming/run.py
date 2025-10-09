import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    for i in range(10):
        for nv in [3, 6, 10, 20, 30]:
            for mu_eff in [0.01, 0.05, 0.1, 0.5, 1.0][::-1]:
                root = f"/home/mmccraw/dev/data/10-06-25/jamming/{nv}-{mu_eff:.2f}/"
                if not os.path.exists(root):
                    os.makedirs(root)

                radii = generate_bidisperse_radii(1000, 0.5, 1.4)
                which = 'small'
                packing_fraction = 0.6

                # build the initial data and equilibrate it, ideally to a 0-overlap state
                n_duplicates = 1
                mu_effs = [mu_eff] * n_duplicates
                nvs = [nv] * n_duplicates
                
                cap_nv = 3
                add_core = True
                rb = build_rigid_bumpy_system_from_radii(radii, which, mu_effs, nvs, packing_fraction, add_core, cap_nv, 'uniform', n_duplicates)
                rb.calculate_mu_eff()
                jamming_data_path = os.path.join(root, f"jamming_{i}")
                rb.set_neighbor_method(NeighborMethod.Cell)
                set_standard_cell_list_parameters(rb, 0.3)
                rb.save(jamming_data_path)

                subprocess.run([
                    os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_jam_pbc"),
                    jamming_data_path,
                    jamming_data_path,
                ], check=True)

                exit()