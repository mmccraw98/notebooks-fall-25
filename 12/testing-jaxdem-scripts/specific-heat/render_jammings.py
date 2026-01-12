import jaxdem as jd

import numpy as np
import subprocess
from pathlib import Path
import h5py

import os
data_root = '/home/mmccraw/dev/data/01-01-26/specific-heat'

if __name__ == "__main__":
    phi = 0.4
    dim = 2
    min_nv = 20
    N = 50

    jamming_root = os.path.join(data_root, 'jamming')

    for mu_eff in [0.01, 0.1, 1.0]:
        for aspect_ratio in [1.0, 1.5, 2.0]:
            particle_name = f'mu-{mu_eff}-alpha-{aspect_ratio}'
            particle_root = os.path.join(jamming_root, particle_name)
            render_name = f"figures/jamming/{particle_name}.png"

            if not os.path.exists(particle_root):
                continue
            if os.path.exists(render_name):
                continue
            
            state = jd.utils.h5.load(os.path.join(particle_root, 'state.h5'))
            system = jd.utils.h5.load(os.path.join(particle_root, 'system.h5'))

            with h5py.File('config.h5', 'w') as f:
                f.create_dataset("pos", data=np.asarray(state.pos))
                f.create_dataset("rad", data=np.asarray(state.rad))
                f.create_dataset("ID",  data=np.asarray(state.ID))
                f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
            script_dir = Path(__file__).resolve().parent
            run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
            subprocess.run([
                str(run_render),
                "config.h5",
                render_name,
                "1000",
            ], check=True)
            os.remove("config.h5")