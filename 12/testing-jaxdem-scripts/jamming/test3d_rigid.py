import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils


dt_min = 1e-3
dt_max = 1e-1

dts = np.logspace(np.log10(dt_min), np.log10(dt_max), 5)

fluctuation = np.zeros_like(dts)
for j, dt in enumerate(dts):
    pos = jnp.asarray(
        [
            # [0.0, 1.0],
            # [1.0, 1.0],
            # [2.0, 1.0],
            [0.0, 1.0, 0],
            [1.0, 1.0, 0],
            [2.0, 1.0, 0],
            [2.0, 3.5, 0],
        ]
    )
    vel = (
        jnp.asarray(
            [
                # [0.0, 1.5, 0],
                # [0.0, 1.5, 0],
                # [0.0, 1.5, 0],
                [0.0, 1.5, 0],
                [0.0, 1.5, 0],
                [0.0, 1.5, 0],
                [0.0, -1.5, 0],
            ]
        )
        * 4
    )
    angVel = jnp.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 0]]) * 1
    ID = jnp.asarray([0, 0, 0, 1])

    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0, poisson=0.3)],
    )

    state = jdem.State.create(pos=pos, ID=ID, vel=vel, angVel=angVel, mat_table=mat_table)


    pos = jnp.asarray(
        [
            [0.0, 1.0, 0],
            [1.0, 1.0, 0],
            [2.0, 1.0, 0],
        ]
    )
    angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
    ID = jnp.asarray([0, 0, 0])
    state = state.add(state, pos=pos - jnp.asarray([0.0, 3.0, 0.0]), angVel=angVel, ID=ID)


    pos = jnp.asarray(
        [
            [1.0, 0.0, 0],
            [1.0, 1.0, 0],
            [2.0, 0.0, 0],
        ]
    )
    angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
    ID = jnp.asarray([0, 0, 0])
    state = state.add(state, pos=pos - jnp.asarray([0.0, 3.0, -4.0]), angVel=angVel, ID=ID)

    pos = jnp.asarray(
        [
            [1.0, 0.0, 0],
            [1.0, 1.0, 0],
            [2.0, 0.0, 0],
            [3.0, 1.0, 0],
            [4.0, 0.0, 0],
        ]
    )
    angVel = jnp.asarray([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]) * 1
    ID = jnp.asarray([0, 0, 0, 0, 0])
    disp = jnp.asarray([0.0, 3.0, 4.0])
    state = state.add(state, pos=pos - disp[None, :], angVel=angVel, ID=ID)

    state.vel /= 100
    state.angVel /= 100


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

    n_steps = int(100000 * dt_min / dt)
    save_stride = 1
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
    fluctuation[j] = np.std(ke + pe) / np.mean(ke + pe)
    plt.plot(ke_t, label='ke trans')
    plt.plot(ke_r, label='ke rot')
    plt.plot(pe, label='pe')
    plt.plot(ke_t + ke_r + pe, label='te')
    plt.legend()
    plt.savefig(f'energies/dt{dt}.png')
    plt.close()

plt.plot(dts, fluctuation)
plt.plot(dts, dts ** 2)
plt.xscale('log')
plt.yscale('log')
plt.savefig('fluc.png')
plt.close()