import pydpmd as dp
from pydpmd.utils import join_systems
import numpy as np
import os
import subprocess

if __name__ == "__main__":
    root = '/Users/marshallmccraw/Projects/yale/data/f-25/12-01-25/grace-data/11-16-25/dynamics/rb-small/T-1.000e-05/'

    T_0 = 1e-5
    T_f = 2e-5
    n_T = 50

    for mu_root in os.listdir(root):
        for offset_root in os.listdir(os.path.join(root, mu_root)):
            path = os.path.join(root, mu_root, offset_root)
            data = dp.data.load(path, location=['final', 'init'])


            # save, set cell list
            # subprocess - minimize energies
            # load data

            for name in ['angular_isf', 'msd', 'g', 'r', 't', 'isf']:
                del data.final.arrays[name]

            temps = np.array([np.ones(data.n_systems()) * T for T in np.linspace(T_0, T_f, n_T)])
            data = join_systems([data for _ in range(n_T)])
            data.set_velocities(temps, np.random.randint(0, 1e9))

            break
        break