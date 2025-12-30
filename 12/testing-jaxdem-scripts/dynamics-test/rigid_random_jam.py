import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils
import numpy as np
import matplotlib.pyplot as plt

def make_grid_state(n_per_axis, dim):
    radius = 0.5
    spacing = 2 * radius
    mass = 1.0
    spacing = jnp.array([spacing for _ in range(dim)])
    state = jdem.utils.gridState.grid_state(
        n_per_axis=[n_per_axis for _ in range(dim)],
        spacing=spacing,
        radius=radius,
        mass=mass,
        jitter=0.0,
    )
    box_size = jnp.max(state.pos, axis=0) + spacing
    return state, box_size

def create_state(dim, n_per_axis, dt, extension, NV):
    sphere_state, box_size = make_grid_state(n_per_axis, dim)
    ID = jnp.concatenate([np.ones(NV) * i for i in sphere_state.ID]).astype(sphere_state.ID.dtype)
    _, nv = jnp.unique(ID, return_counts=True)
    local_id = jnp.arange(ID.size) - jnp.concatenate((jnp.zeros(1), jnp.cumsum(nv))).astype(sphere_state.ID.dtype)[ID]
    orientation = 2 * np.pi * local_id / nv[ID]
    pos_c = sphere_state.pos.copy()[ID]
    rad = (sphere_state.rad / nv)[ID]
    mass = (sphere_state.mass / nv)[ID]
    if sphere_state.dim == 2:
        angles = jnp.column_stack((jnp.cos(orientation), jnp.sin(orientation)))

    else:
        angles = jnp.column_stack((jnp.cos(orientation), jnp.sin(orientation), jnp.zeros_like(orientation)))
    pos_p = (sphere_state.rad[ID] * extension - rad)[:, None] * angles

    pos_lab = pos_p + pos_c

    state = jdem.State.create(pos=pos_lab, rad=rad, ID=ID)

    mats = [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=0.27)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    system = jdem.System.create(
        state.shape,
        dt=dt,
        collider_type="naive",
        domain_type="periodic",
        rotation_integrator_type="verletspiral",
        linear_integrator_type="verlet",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size
        ),
    )
    state = jdem.utils.compute_clump_properties(state, system.mat_table, n_samples=50_000)

    return state, system

# assign clump properties based on the total mass in the particle

dt = 1e-2
n_steps = 1_000_0
save_stride = 100
n_snapshots = n_steps // save_stride

dim = 3
n_per_axis = 2
NV = 3
extension = 0.6

state, system = create_state(dim, n_per_axis, dt, extension, NV)


seed = np.random.randint(0, 1000000)
key = jax.random.PRNGKey(seed)
key_vel, key_angVel = jax.random.split(key, 2)
cid, offsets = jnp.unique(state.ID, return_index=True)
N_clumps = cid.size
clump_vel = jax.random.normal(key_vel, (N_clumps, state.dim))
clump_vel -= jnp.mean(clump_vel, axis=0)
state.vel = clump_vel[state.ID]
ke_t = jnp.sum(0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1))
# ke_r ?????
dof = (state.dim + state.inertia.shape[-1]) * N_clumps - state.dim
current_temp = 2 * ke_t / dof
scale = jnp.sqrt(1e-4 / current_temp)
state.vel *= scale

state, syste, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

state, system, phi, pe = jdem.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-4)

import subprocess
from pathlib import Path
import h5py
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
    "jammed.png",
    "1000",
], check=True)

# H5 WRITER
# CALCULATE ENERGY AND FORCE - USE IN MINIMIZER