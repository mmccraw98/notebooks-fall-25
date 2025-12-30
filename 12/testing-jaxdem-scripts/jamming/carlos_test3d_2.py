import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jdem
import jaxdem.utils as utils
import numpy as np
import matplotlib.pyplot as plt

def make_grid_state(n_per_axis, dim):
    radius = 0.5
    spacing = 3 * radius
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

def create_state(dt=0.001):
    n_per_axis=2
    dim=3


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

    true_mass = jnp.ones_like(state.mass) * 1.0
    state.inertia *= (true_mass / state.mass)[..., None]
    state.mass = true_mass
    print(state.mass)

    return state, system

# assign clump properties based on the total mass in the particle

E_std = []
dts = 10 ** np.linspace(-4, -1, 5)
dt_min = np.min(dts)

for dt in dts:
    # dt = 0.001
    n_steps = int(10000000 * dt_min / dt)
    save_stride = 100
    n_snapshots = n_steps // save_stride

    state, system = create_state(dt)


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
    scale = jnp.sqrt(1e-3 / current_temp)
    state.vel *= scale

    state, syste, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
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

    print(f"dt = {dt:.2e} | steps = {n_snapshots*save_stride:07d}  | E_std = {res:.2e}")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.gca().set_aspect('equal')
    for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
        plt.gca().add_artist(Circle(p, r))
    plt.xlim(0, system.domain.box_size[0])
    plt.ylim(0, system.domain.box_size[1])
    plt.savefig(f'final_state_{dt}.png')
    plt.close()

    plt.plot(E, label="Energy")
    plt.plot(Pe, label="Pe")
    plt.plot(Ke_t, label="Ke_t")
    plt.plot(Ke_r, label="Ke_r")
    plt.legend()
    plt.savefig(f"energy_components_{dt}.png", dpi=400)
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