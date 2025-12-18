import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

def make_grid_state(n_per_axis, dim, jitter=0):
    radius = 0.5
    spacing = 3 * radius
    mass = 1.0
    spacing = jnp.array([spacing for _ in range(dim)])
    state = jdem.utils.gridState.grid_state(
        n_per_axis=[n_per_axis for _ in range(dim)],
        spacing=spacing,
        radius=radius,
        mass=mass,
        jitter=jitter,
    )
    box_size = jnp.max(state.pos, axis=0) + spacing
    return state, box_size

def create_state(dt=0.001):
    n_per_axis=3
    dim=3


    sphere_state, box_size = make_grid_state(n_per_axis, dim, jitter=5e-1)
    NV = 2
    ID = jnp.concatenate([np.ones(NV) * i for i in sphere_state.ID]).astype(sphere_state.ID.dtype)
    _, nv = jnp.unique(ID, return_counts=True)
    local_id = jnp.arange(ID.size) - jnp.concatenate((jnp.zeros(1), jnp.cumsum(nv))).astype(sphere_state.ID.dtype)[ID]
    random_angle = np.random.rand(nv.size) * 2 * np.pi
    orientation = 2 * np.pi * local_id / nv[ID] + random_angle[ID]
    pos_c = sphere_state.pos.copy()[ID]
    # rad = (sphere_state.rad / nv)[ID]
    rad = (sphere_state.rad * 0.6)[ID]
    mass = (sphere_state.mass / nv)[ID]
    if sphere_state.dim == 2:
        angles = jnp.column_stack((jnp.cos(orientation), jnp.sin(orientation)))

    else:
        angles = jnp.column_stack((jnp.cos(orientation), jnp.sin(orientation), jnp.zeros_like(orientation)))
    pos_p = (sphere_state.rad[ID] - rad)[:, None] * angles

    pos_lab = pos_p + pos_c

    state = jdem.State.create(pos=pos_lab, rad=rad, ID=ID)

    mats = [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=0.27)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    system = jdem.System.create(
        state.shape,
        dt=dt,
        collider_type="naive",
        # collider_type="celllist",
        # collider_kw=dict(state=state),
        domain_type="periodic",
        # rotation_integrator_type="rotationgradientdescent",
        # rotation_integrator_kw=dict(learning_rate=1e-6),
        # linear_integrator_type="lineargradientdescent",
        # rotation_integrator_type="",
        rotation_integrator_type="rotationfire",
        linear_integrator_type="linearfire",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size
        ),
    )
    state = jdem.utils.compute_clump_properties(state, system.mat_table, n_samples=50_000)

    return state, system

# assign clump properties based on the total mass in the particle

state, system = create_state(1e-2)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.gca().set_aspect('equal')
for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
    plt.gca().add_artist(Circle(p, r))
plt.xlim(0, system.domain.box_size[0])
plt.ylim(0, system.domain.box_size[1])
plt.savefig(f'init_state.png')
plt.close()

import h5py
with h5py.File('init_config.h5', 'w') as f:
    f.create_dataset("pos", data=np.asarray(state.pos))
    f.create_dataset("rad", data=np.asarray(state.rad))
    f.create_dataset("ID",  data=np.asarray(state.ID))
    f.create_dataset("box_size", data=np.asarray(system.domain.box_size))

script_dir = Path(__file__).resolve().parent
run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
subprocess.run([
    str(run_render),
    "init_config.h5",
    "init_config_render.png",
    "1000",
], check=True)

state, system, phi, pe = jdem.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-4)
print(phi, pe)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.gca().set_aspect('equal')
for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
    plt.gca().add_artist(Circle(p, r))
plt.xlim(0, system.domain.box_size[0])
plt.ylim(0, system.domain.box_size[1])
plt.savefig(f'final_state.png')
plt.close()

import h5py
with h5py.File('final_config.h5', 'w') as f:
    f.create_dataset("pos", data=np.asarray(state.pos))
    f.create_dataset("rad", data=np.asarray(state.rad))
    f.create_dataset("ID",  data=np.asarray(state.ID))
    f.create_dataset("box_size", data=np.asarray(system.domain.box_size))

script_dir = Path(__file__).resolve().parent
run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
subprocess.run([
    str(run_render),
    "final_config.h5",
    "final_config_render.png",
    "1000",
], check=True)