import pydpmd as md
import numpy as np
import os
from system_building_resources import *
import time
import subprocess
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/10-16-25/data-from-grace/rb-compression-diffusion-lower-temp'

    for dirpath, dirnames, fnames in tqdm(os.walk(root)):
        if 'meta.h5' in fnames and 'jamming_' in os.path.basename(dirpath):
            calculation_path = dirpath.replace('jamming', 'calculation')

            data = load(dirpath, location=['final', 'init'])
            data.set_neighbor_method(NeighborMethod.Cell)
            set_standard_cell_list_parameters(data, 0.3)
            data.save(calculation_path)

            subprocess.run([
                os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_calculate"),
                calculation_path,
                calculation_path,
            ], check=True)
