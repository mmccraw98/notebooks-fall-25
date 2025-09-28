import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time

if __name__ == "__main__":
    temperature = 1e-3
    input_data_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_prelim"
    output_data_path = "/home/mmccraw/dev/data/09-09-25/new-initializations/rb_dynamics"

    rb = load(input_data_path, location=["final", "init"])
    rb.set_velocities(temperature, int(time.time()))
    rb.save(output_data_path)

    # subprocess.run([
    #     os.path.join("/home/mmccraw/dev/dpmd/build/", "nve_rigid_bumpy_pbc_final"),
    #     output_data_path,
    #     output_data_path,
    #     str(1e4),
    #     str(1e0)
    # ], check=True)

    phi_increment = 1e-2
    compression_steps = 1e4
    damping_scale = 1e1
    compression_frequency = int(compression_steps * 1e-3 / phi_increment)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "damped_dynamics_compress_to_packing_fraction"),
        output_data_path,
        output_data_path,
        str(compression_steps),
        str(1e4),
        str(damping_scale),
        str(compression_frequency),
        str(temperature)
    ], check=True)