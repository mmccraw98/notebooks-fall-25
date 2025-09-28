import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

if __name__ == "__main__":
    input_data_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_prelim"
    
    n_dynamics_steps = 1e6
    save_frequency = 1e2
    temperature = 1e-3
    phi_increment = 1e-2
    compression_steps = 1e4
    damping_scale = 1e1
    compression_frequency = int(compression_steps * 1e-3 / phi_increment)
    packing_fraction_target = 0.8

    i = 0
    while True:
        compression_data_path = f"/home/mmccraw/dev/data/09-09-25/new-initializations/compression_{i}"
        dynamics_data_path = f"/home/mmccraw/dev/data/09-09-25/new-initializations/dynamics_{i}"

        rb = load(input_data_path, location=["final", "init"])
        rb.set_velocities(temperature, int(time.time()))
        rb.save(compression_data_path)

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "damped_dynamics_compress_to_packing_fraction"),
            compression_data_path,
            compression_data_path,
            str(compression_steps),
            str(1e5),
            str(damping_scale),
            str(compression_frequency),
            str(temperature)
        ], check=True)

        rb = load(compression_data_path, location=["final", "init"])
        if np.any(rb.final.packing_fraction >= packing_fraction_target):
            break
        rb.set_velocities(temperature, int(time.time()))
        rb.set_neighbor_method(NeighborMethod.Cell)
        set_standard_cell_list_parameters(rb, 0.3)
        rb.save(dynamics_data_path)

        subprocess.run([
            os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
            dynamics_data_path,
            dynamics_data_path,
            str(n_dynamics_steps),
            str(save_frequency)
        ], check=True)

        input_data_path = dynamics_data_path

        i += 1
