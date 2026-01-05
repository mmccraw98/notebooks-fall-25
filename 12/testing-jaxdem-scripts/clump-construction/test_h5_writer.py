from typing import Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import warnings
import jax
import jax.numpy as jnp
import jaxdem as jd

jax.config.update("jax_enable_x64", True)

N = 50
phi = 0.4
dim = 2

particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
sphere_pos, box_size = jd.utils.random_sphere_configuration(particle_radii, phi, dim)

state = jd.State.create(
    pos=sphere_pos,
    rad=particle_radii,
)

mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

system = jd.System.create(
    state_shape=state.shape,
    dt=1e-2,
    # linear_integrator_type="verlet",
    # rotation_integrator_type="verletspiral",
    linear_integrator_type="linearfire",
    rotation_integrator_type="rotationfire",
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    # collider_type="celllist",
    # collider_kw=dict(state=state),
    mat_table=mat_table,
    domain_kw=dict(
        box_size=box_size,
    ),
)

jd.utils.h5.save(state, 'state.h5')
jd.utils.h5.save(system, 'system.h5')
new_state = jd.utils.h5.load('state.h5')
new_system = jd.utils.h5.load('system.h5')


