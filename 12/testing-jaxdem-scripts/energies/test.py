import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import jaxdem as jd
import numpy as np
import os
import h5py

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

phi = 0.6
dim = 2
N = 100
e_int = 1.0
dt = 1e-2
target_temperatures = np.logspace(-5, -2, 10)
particle_radii = []
for _ in range(target_temperatures.size):
    particle_radii.append(jd.utils.dispersity.get_polydisperse_radii(N))
pos, box_size = random_sphere_configuration(particle_radii, phi, dim)
cutoff = jnp.max(jnp.array(particle_radii))
particle_radii = jnp.array(particle_radii)

def _create(i):
    state = jd.State.create(
        pos=pos[i],
        rad=particle_radii[i],
        mass=jnp.ones(pos[i].shape[0])
    )
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=cutoff
        ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system

def compute_ke(state):
    return 0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1)

def compute_temp(state):
    dof = (state.N - 1) * state.dim
    ke = compute_ke(state)
    return 2 * jnp.sum(ke, axis=-1) / dof

def scale_temps(state, target_temperatures):
    state.vel -= jnp.mean(state.vel, axis=-2, keepdims=True)
    temperature = compute_temp(state)
    scale = jnp.sqrt(target_temperatures / temperature)
    state.vel *= scale[:, None, None]
    return state

state, system = jax.vmap(_create)(jnp.arange(target_temperatures.size))
key = jax.random.PRNGKey(np.random.randint(0, 1e9))
state.vel = jax.random.normal(key, state.vel.shape)
state = scale_temps(state, target_temperatures)


# (Dynamic) Cell List: better for clumps
# Static Cell List: better for spheres

# fragility of spheres
# isf
# g(r)
# msd
# aisf
# msad

# install mypylinter
# metrics object as a pointer inside of system
# flax (jax neural network api) has a metric object

# traffic paper