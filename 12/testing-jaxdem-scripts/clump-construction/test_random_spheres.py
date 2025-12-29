from typing import Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import warnings
import jax
import jax.numpy as jnp
import jaxdem as jd

N = 50
phi = 0.4
dim = 2

particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
sphere_pos, box_size = jd.utils.random_sphere_configuration(particle_radii, phi, dim)

print(sphere_pos)
