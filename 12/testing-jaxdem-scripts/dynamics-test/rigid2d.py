import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils


dt = 1e-2

pos = jnp.asarray(
    [
        # [0.0, 1.0],
        # [1.0, 1.0],
        # [2.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 3.5],
    ]
)
vel = (
    jnp.asarray(
        [
            # [0.0, 1.5, 0],
            # [0.0, 1.5, 0],
            # [0.0, 1.5, 0],
            [0.0, 1.5],
            [0.0, 1.5],
            [0.0, 1.5],
            [0.0, -1.5],
        ]
    )
    * 4
)
angVel = jnp.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]]) * 1
ID = jnp.asarray([0, 0, 0, 1])

mat_table = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", density=0.27, young=1.0e4, poisson=0.3)],
)

state = jdem.State.create(pos=pos, ID=ID, vel=vel, mat_table=mat_table)


pos = jnp.asarray(
    [
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ]
)
angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
ID = jnp.asarray([0, 0, 0])
state = state.add(state, pos=pos - jnp.asarray([0.0, 3.0]), ID=ID)


# pos = jnp.asarray(
#     [
#         [1.0, 0.0],
#         [1.0, 1.0],
#         [2.0, 0.0],
#     ]
# )
# angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
# ID = jnp.asarray([0, 0, 0])
# state = state.add(state, pos=pos - jnp.asarray([0.0, 3.0]), ID=ID)

# pos = jnp.asarray(
#     [
#         [1.0, 0.0],
#         [1.0, 1.0],
#         [2.0, 0.0],
#         [3.0, 1.0],
#         [4.0, 0.0],
#     ]
# )
# angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
# ID = jnp.asarray([0, 0, 0, 0, 0])
# disp = jnp.asarray([0.0, 3.0])
# state = state.add(state, pos=pos - disp[None, :], ID=ID)


system = jdem.System.create(
    state.shape,
    dt=dt,
    collider_type="naive",
    # collider_kw=dict(state=state),
    domain_type="periodic",
    linear_integrator_type="verlet",
    rotation_integrator_type="spiral",
    domain_kw=dict(
        box_size=10.0 * jnp.ones(state.dim), anchor=-5 * jnp.ones(state.dim)
    ),
)

state = utils.compute_clump_properties(state, system.mat_table, n_samples=50_000)

print(state.mass)

n_steps = 1_000_000
save_stride = 100
n_snapshots = n_steps // save_stride
final_state, final_system, (traj_state, traj_system) = jdem.System.trajectory_rollout(
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


import matplotlib.pyplot as plt
plt.plot(pe, label='pe')
plt.plot(ke_t, label='ke_t')
plt.plot(ke_r, label='ke_r')
plt.plot(ke_r + ke_t + pe, label='te')
plt.legend()
plt.savefig('energies2d.png')
plt.close()

import subprocess
from pathlib import Path
import h5py
with h5py.File("traj.h5", "w") as f:
    f.create_dataset("pos", data=np.asarray(traj_state.pos))
    f.create_dataset("rad", data=np.asarray(traj_state.rad))
    f.create_dataset("ID", data=np.asarray(traj_state.ID))
    f.create_dataset("box_size", data=np.asarray(traj_system.domain.box_size))

# --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
script_dir = Path(__file__).resolve().parent
run_animation = script_dir.parent / "animation" / "run_animation.sh"
subprocess.run(
    [
        str(run_animation),
        "traj.h5",
        "traj2d.gif",
        "100",   # num_frames (evenly sampled if traj has more)
        "1000",  # base_pixels
        "15",    # fps
    ],
    check=True,
)