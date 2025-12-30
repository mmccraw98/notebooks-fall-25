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

def create_state(n_per_axis, dt, dim):
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


    # solve for the mass, com, inertia, and orientation

    clump_mass = jax.ops.segment_sum(mass, ID, num_segments=nv.size)
    clump_com_pos = jax.ops.segment_sum(mass[:, None] * pos_lab, ID, num_segments=nv.size) / clump_mass[:, None]

    r = pos_lab - clump_com_pos[ID]

    # 2d
    if dim == 2:
        clump_inertia = jax.ops.segment_sum(mass * jnp.sum(r * r, axis=-1), ID, num_segments=nv.size)[:, None]
        cov = jax.ops.segment_sum(mass[:, None, None] * (r[:, :, None] * r[:, None, :]), ID, num_segments=nv.size)
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        v = eigvecs[:, :, 1]
        theta = jnp.arctan2(v[:, 1], v[:, 0])
        half = 0.5 * theta
        q_wxyz = jnp.stack([jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)], axis=-1)
    else:
        I_tensor = mass[:, None, None] * (jnp.sum(r * r, axis=-1)[:, None, None] * jnp.eye(3)[None, :, :] - (r[:, :, None] * r[:, None, :]))
        I_tensor = 0.5 * (I_tensor + jnp.swapaxes(I_tensor, -1, -2))  # symmetrize
        eigvals, eigvecs = jnp.linalg.eigh(I_tensor)
        clump_inertia = eigvals
        det = jnp.linalg.det(eigvecs)
        eigvecs = jnp.where(det[:, None, None] < 0, eigvecs.at[:, :, 0].multiply(-1), eigvecs)
        q_xyzw = jax.vmap(lambda Rm: Rotation.from_matrix(Rm).as_quat())(eigvecs)
        q_wxyz = jnp.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=-1)

    pos_c = clump_com_pos[ID]
    inertia = clump_inertia[ID]
    q = Quaternion(q_wxyz[ID, 0:1], q_wxyz[ID, 1:4])
    pos_p = Quaternion.rotate_back(q, pos_lab - pos_c)

    assert jnp.all(jnp.isclose(Quaternion.rotate(q, pos_p), pos_lab - pos_c))  # verify it has been done correctly

    state = jd.State(
        pos_c=pos_c,
        pos_p=pos_p,
        q=q,
        rad=rad,
        volume=jnp.ones_like(rad),
        mass=mass,
        inertia=inertia,
        ID=ID,
        vel=jnp.zeros_like(pos_c, float),
        angVel=jnp.zeros_like(inertia, float),
        force=jnp.zeros_like(pos_c, float),
        torque=jnp.zeros_like(inertia, float),
        mat_id=jnp.zeros_like(rad, int),
        species_id=jnp.zeros_like(rad, int),
        fixed=jnp.zeros_like(rad, bool),
    )

    system = jd.System.create(
        state.shape,
        dt=dt,
        collider_type="naive",
        # collider_kw=dict(state=state),
        domain_type="periodic",
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_kw=dict(
            box_size=box_size
        ),
    )

    seed = np.random.randint(0, 1000000)
    key = jax.random.PRNGKey(seed)
    key_vel, key_angVel = jax.random.split(key, 2)
    cid, offsets = jnp.unique(state.ID, return_index=True)
    N_clumps = cid.size
    state.vel = jax.random.normal(key_vel, state.vel.shape) * 1e-2
    state.vel -= jnp.mean(state.vel, axis=0)

    return state, system


E_std = []
dts = 10 ** np.linspace(-4.5, -2.5, 6)

n_per_axis = 4
dim = 3
dt = 1e-2

for dt in dts:
    # dt = 0.001
    time = 12.0
    frames = 200
    stride = int(time / dt) // frames
    state, system = create_state(n_per_axis, dt, dim)

    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=frames, stride=stride
    )

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
    # plt.savefig("energy_conservation.png", dpi=400)
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


# print(state.vel)

# n_steps = 100000
# save_stride = 100
# n_snapshots = n_steps // save_stride
# final_state, final_system, (traj_state, traj_system) = jd.System.trajectory_rollout(
#     state, system, n=n_snapshots, stride=save_stride
# )

# _, offsets = jnp.unique(state.ID, return_index=True)

# pe = jnp.sum(
#     jax.vmap(
#         lambda st, sys:
#         sys.collider.compute_potential_energy(st, sys))(traj_state, traj_system)[:, offsets],
#     axis=-1
# )
# ke_t = jnp.sum((0.5 * traj_state.mass * jnp.vecdot(traj_state.vel, traj_state.vel))[:, offsets], axis=-1)
# w = traj_state.q.rotate_back(traj_state.q, traj_state.angVel)
# ke_r = jnp.sum((0.5 * jnp.vecdot(w, traj_state.inertia * w))[:, offsets], axis=-1)
# ke = ke_t + ke_r
# plt.plot(ke_t, label='ke trans')
# plt.plot(ke_r, label='ke rot')
# plt.plot(pe, label='pe')
# plt.plot(ke_t + ke_r + pe, label='te')
# plt.legend()
# plt.savefig(f'test.png')
# plt.close()