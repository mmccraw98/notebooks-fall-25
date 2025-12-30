import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils
import numpy as np
import matplotlib.pyplot as plt


def create_state(dt=0.001):
    pos = jnp.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 3.5, 0.0],
            [0.0, -2.0, 0.0],
            [1.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [1.0, -3.0, 4.0],
            [1.0, -2.0, 4.0],
            [2.0, -3.0, 4.0],
            [1.0, -3.0, -4.0],
            [1.0, -2.0, -4.0],
            [2.0, -3.0, -4.0],
            [3.0, -2.0, -4.0],
            [4.0, -3.0, -4.0],
        ]
    )
    vel = jnp.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    angVel = (
        jnp.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
        / 2
    )
    ID = jnp.asarray([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4])
    state = jdem.State.create(pos=pos, vel=vel, angVel=angVel, ID=ID)

    system = jdem.System.create(
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
    state = utils.compute_clump_properties(state, system.mat_table, n_samples=50_000)
    return state, system


E_std = []
dts = 10 ** np.linspace(-5, -2, 6)

for dt in dts:
    # dt = 0.001
    time = 10.0
    frames = 200
    stride = int(time / dt) // frames
    state, system = create_state(dt)

    state, syste, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=frames, stride=stride
    )

    writer = jdem.VTKWriter()
    writer.save(state_traj, system_traj, trajectory=True)

    _, indices = jnp.unique(state.ID, return_index=True)
    Ke_t = 0.5 * jnp.sum(
        (state_traj.mass * jnp.vecdot(state_traj.vel, state_traj.vel))[:, indices],
        axis=1,
    )
    angVel = state_traj.q.rotate_back(state_traj.q, state_traj.angVel)
    Ke_r = 0.5 * jnp.sum(
        jnp.vecdot(angVel, state_traj.inertia * angVel)[:, indices], axis=1
    )
    Pe = jnp.sum(
        jax.vmap(system_traj.collider.compute_potential_energy)(
            state_traj, system_traj
        ),
        axis=1,
    )
    E = Ke_t + Ke_r + Pe
    res = jnp.std(E)
    E_std.append(res)

    print(f"dt = {dt:.2e} | steps = {frames*stride:07d}  | E_std = {res:.2e}")

    plt.plot(E, label="Energy")
    plt.plot(Pe, label="Pe")
    plt.plot(Ke_t, label="Ke_t")
    plt.plot(Ke_r, label="Ke_r")
    plt.legend()
    plt.savefig(f"energy_conservation_{dt}.png", dpi=400)
    plt.close()


from scipy.stats import linregress


log_dts = np.log10(dts)
log_E_std = np.log10(np.array(E_std))

slope, intercept, r_value, p_value, std_err = linregress(log_dts, log_E_std)

print("\n--- Linear Fit Results ---")
print(f"Convergence Order (k)   : {slope:.3f}")
print(f"Y-intercept (log10(C))  : {intercept:.3f}")
print(f"R-squared value         : {r_value**2:.4f}")

# Reconstruct the fit line for plotting
fit_line = 10 ** (intercept + slope * log_dts)

plt.plot(dts, E_std, label="Energy")
plt.plot(dts, fit_line, "-", label=f"Fit Line: $\\text{{dt}}^{{{slope:.3f}}}$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("energy_conservation.png", dpi=400)
plt.show()