import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import shutil
from correlation_functions import compute_shear_modulus


if __name__ == "__main__":
    for root in [
        # "/home/mmccraw/dev/data/10-15-25/new-shear-modulus/trial-0/",
        '/home/mmccraw/dev/data/10-15-25/new-shear-modulus-dilute/trial-0/',
        '/home/mmccraw/dev/data/10-15-25/new-shear-modulus-full-range/trial-0/'
    ]:
        for path in os.listdir(root):
            if 'minimization_' not in path:
                continue

            file_index = path.split('_')[-1]
            torques_path = os.path.join(root, f'torques_{file_index}')
            
            rb = load(os.path.join(root, path), location=['init'])
            rb.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(rb, 0.3)
            rb.save(torques_path)
            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
                torques_path,
                torques_path,
                str(1e4),  # total number of dynamics steps
                str(1e1),  # save frequency
                str(1e-2),
            ], check=True)

            # calculate the shear modulus and save it with mu_eff, nv, packing_fraction, and temperature
            rb = load(torques_path, location=["final", "init"], load_trajectory=True, load_full=True)
            rb.calculate_mu_eff()
            shutil.rmtree(torques_path)
            counts = np.add.reduceat(np.sum(np.abs(rb.trajectory.torque) > 0, axis=0), rb.system_offset[:-1])
            torques = np.add.reduceat(np.sum(np.abs(rb.trajectory.torque), axis=0), rb.system_offset[:-1])
            np.savez(
                torques_path.rstrip('/') + '.npz',
                avg_torque=torques / counts,
                mu_eff=rb.mu_eff[rb.system_offset[:-1]],
                nv=rb.n_vertices_per_particle[rb.system_offset[:-1]],
                packing_fraction=rb.init.packing_fraction,
            )