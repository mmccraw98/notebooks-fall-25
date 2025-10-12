import pydpmd as dp
import numpy as np
import os
from system_building_resources import *
import time
import shutil

if __name__ == "__main__":
    in_dir = '/home/mmccraw/dev/data/10-11-25/effective-potential/jamming/rb-10-0.05/jamming_0/'
    out_dir = '/home/mmccraw/dev/data/10-11-25/effective-potential/jamming/rb-10-0.05/calculation_0/'

    rb = dp.data.load(in_dir, location=['final', 'init'])
    rb.calculate_mu_eff()
    rb.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(rb, 0.3)
    rb.save(out_dir)

    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_calculate"),
        out_dir,
        out_dir,
    ], check=True)