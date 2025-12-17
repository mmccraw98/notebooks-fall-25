import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import replace
from functools import partial

import jaxdem as jd
from jaxdem.utils.quaternion import Quaternion
from jax.scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def make_grid_state(n_per_axis, dim):
    radius = 0.5
    spacing = 3 * radius
    mass = 1.0
    spacing = jnp.array([spacing for _ in range(dim)])
    state = jd.utils.gridState.grid_state(
        n_per_axis=[n_per_axis for _ in range(dim)],
        spacing=spacing,
        radius=radius,
        mass=mass,
        jitter=0.0,
    )
    box_size = jnp.max(state.pos, axis=0) + spacing
    return state, box_size

def create_state(n_per_axis, dt, dim, temp):
    sphere_state, box_size = make_grid_state(n_per_axis, dim)
    NV = 2
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
    pos_p = (sphere_state.rad[ID] - rad)[:, None] * angles

    pos_lab = pos_p + pos_c

    state = jd.State.create(pos=pos_lab, ID=ID)
    system = jd.System.create(
        state.shape,
        dt=dt,
        collider_type="naive",
        domain_type="periodic",
        rotation_integrator_type="verletspiral",
        linear_integrator_type="verlet",
        domain_kw=dict(
            box_size=10.0 * jnp.ones(state.dim), anchor=-5 * jnp.ones(state.dim)
        ),
    )
    state = jd.utils.compute_clump_properties(state, system.mat_table, n_samples=50_000)
    print(state.inertia.shape)

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
    scale = jnp.sqrt(temp / current_temp)
    state.vel *= scale
    return state, system

n_per_axis = 4
dt = 1e-2
dim = 2
temp = 1e-4

state, system = create_state(n_per_axis, dt, dim, temp)

n_steps = 100000
save_stride = 100
n_snapshots = n_steps // save_stride
final_state, final_system, (traj_state, traj_system) = jd.System.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

_, offsets = jnp.unique(state.ID, return_index=True)

pe = jnp.sum(
    jax.vmap(
        lambda st, sys:
        sys.collider.compute_potential_energy(st, sys))(traj_state, traj_system)[:, offsets],
    axis=-1
)
ke_t = jnp.sum((0.5 * traj_state.mass * jnp.vecdot(traj_state.vel, traj_state.vel))[:, offsets], axis=-1)
w = traj_state.q.rotate_back(traj_state.q, traj_state.angVel)
ke_r = jnp.sum((0.5 * jnp.vecdot(w, traj_state.inertia * w))[:, offsets], axis=-1)
ke = ke_t + ke_r
plt.plot(ke_t, label='ke trans')
plt.plot(ke_r, label='ke rot')
plt.plot(pe, label='pe')
plt.plot(ke_t + ke_r + pe, label='te')
plt.legend()
plt.savefig(f'test.png')
plt.close()