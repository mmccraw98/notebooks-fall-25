import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
from correlation_functions import compute_msd, compute_shear_modulus, compute_rotational_msd, compute_pair_correlation_function, compute_vacf, compute_rotational_msd
from pydpmd.calc import run_binned, run_binned_ragged, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
from tqdm import tqdm
from scipy.optimize import minimize
import pandas as pd
from correlation_functions import compute_neighbor_list_for_all_frames, simple_angle_disp_kernel, angle_disp_kernel, angle_disp_kernel_denominator
import subprocess

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/10-01-25/short-test-3/trial-0/'

    for path in os.listdir(root):
        if not path.startswith('dynamics_'):
            continue

        subprocess.run([
            'python',
            '_run_step.py',
            '--root', root,
            '--path', path
        ])