import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.plot import draw_particles_frame, create_animation, downsample
from correlation_functions import compute_msd, compute_shear_modulus, compute_rotational_msd, compute_pair_correlation_function, compute_angle_pair_correlation_function, compute_angle_simple_pair_correlation_function
from pydpmd.calc import run_binned, fused_msd_kernel, TimeBins, LagBinsExact, LagBinsLog, LagBinsLinear, LagBinsPseudoLog, requires_fields
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re
from tqdm import tqdm


root = '/home/mmccraw/dev/data/09-27-25/run-1/'
R = re.compile(rf'^dynamics_(\d+)$')
ds = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d)) and R.fullmatch(d)]
run_names = [os.path.join(root,d) for d in sorted(ds,key=lambda s:int(R.fullmatch(s).group(1)))]

for run_name in run_names:
    try:
        data = load(run_name, location=['final', 'init'], load_trajectory=True, load_full=False)
        data.calculate_mu_eff()
    except:
        continue
    msd_path = os.path.join(run_name, 'msd.npz')
    msd, t = compute_rotational_msd(data, msd_path, overwrite=False)
    shear_modulus_path = os.path.join(run_name, 'shear_modulus.npz')
    shear_modulus, t = compute_shear_modulus(data, shear_modulus_path, overwrite=False, subtract_mean_stress=True)
    pair_correlation_function_path = os.path.join(run_name, 'pair_correlation_function.npz')
    radial_bins = np.linspace(0.5, 3, 1000)
    G, r = compute_pair_correlation_function(data, radial_bins, pair_correlation_function_path)
    angle_simple_pair_correlation_function_path = os.path.join(run_name, 'angle_simple_pair_correlation_function.npz')
    G_delta, t = compute_angle_simple_pair_correlation_function(data, angle_simple_pair_correlation_function_path)