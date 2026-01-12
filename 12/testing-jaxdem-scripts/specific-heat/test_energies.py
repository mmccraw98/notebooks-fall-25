from typing import Optional, Tuple
import jax.numpy as jnp
import jax
import jaxdem as jd
import numpy as np
import os
import h5py


# (Dynamic) Cell List: better for clumps
# Static Cell List: better for spheres


from jaxdem.utils.quaternion import Quaternion


def _as_batched_temperatures(state, temperature):
    """
    Returns T_full with shape == state.ID.shape[:-1] (lead dims), where lead dim 0 is batch if present.
    Supports: scalar temperature, or array of shape (B,).
    """
    lead_shape = state.ID.shape[:-1]
    if len(lead_shape) == 0:
        return jnp.asarray(temperature, dtype=float).reshape(())

    B = lead_shape[0]
    T = jnp.asarray(temperature, dtype=float)
    if T.ndim == 0:
        Tb = jnp.full((B,), T, dtype=float)
    else:
        assert T.shape == (B,), f"temperature must be scalar or shape (B,), got {T.shape} with B={B}"
        Tb = T

    # broadcast across any additional (trajectory) leading dims after batch
    reshape = (B,) + (1,) * (len(lead_shape) - 1)
    return jnp.broadcast_to(Tb.reshape(reshape), lead_shape)


def _body_weights_from_ids(ID_i, N):
    # ID_i: (N,), IDs in [0, N-1] (State.create enforces that)
    counts = jnp.bincount(ID_i, length=N)  # (N,)
    w = 1.0 / counts[ID_i]                 # (N,)
    return counts, w


def kinetic_energy_translation(state):
    """
    Translational KE per *rigid body* (unique state.ID), not per member-sphere.
    Works for snapshots, batched states, and rolled-out trajectories.
    Returns shape == state.mass.shape[:-1] (all leading dims).
    """
    N, dim = state.N, state.dim
    lead_shape = state.mass.shape[:-1]
    M = int(jnp.prod(jnp.array(lead_shape))) if lead_shape else 1

    ID = state.ID.reshape((M, N))
    mass = state.mass.reshape((M, N))
    fixed = state.fixed.reshape((M, N))
    vel = state.vel.reshape((M, N, dim))

    def _one(ID_i, m_i, fixed_i, v_i):
        counts, w = _body_weights_from_ids(ID_i, N)
        movable = (1.0 - fixed_i).astype(v_i.dtype)

        # body mass/velocity (count each body once via weights)
        m_body = jax.ops.segment_sum(m_i * movable * w, ID_i, num_segments=N)                 # (N,)
        v_body = jax.ops.segment_sum(v_i * (movable * w)[:, None], ID_i, num_segments=N)      # (N,dim)

        # KE = 1/2 sum_b M_b |V_b|^2
        v2 = jnp.sum(v_body * v_body, axis=-1)
        return 0.5 * jnp.sum(m_body * v2)

    ke_flat = jax.vmap(_one)(ID, mass, fixed, vel)
    return ke_flat.reshape(lead_shape) if lead_shape else ke_flat.reshape(())


def kinetic_energy_rotation(state):
    """
    Rotational KE per *rigid body* for clumps only (IDs with multiple members).
    Spheres (unique IDs) contribute zero by definition here.
    Works for snapshots, batched states, and rolled-out trajectories.
    Returns shape == state.mass.shape[:-1] (all leading dims).
    """
    N, dim = state.N, state.dim
    ang_dim = 1 if dim == 2 else 3

    lead_shape = state.mass.shape[:-1]
    M = int(jnp.prod(jnp.array(lead_shape))) if lead_shape else 1

    ID = state.ID.reshape((M, N))
    inertia = state.inertia.reshape((M, N, ang_dim))
    fixed = state.fixed.reshape((M, N))
    angVel = state.angVel.reshape((M, N, ang_dim))
    qw = state.q.w.reshape((M, N, 1))
    qxyz = state.q.xyz.reshape((M, N, 3))

    def _one(ID_i, I_i, fixed_i, w_lab_i, qw_i, qxyz_i):
        counts, w = _body_weights_from_ids(ID_i, N)
        movable = (1.0 - fixed_i).astype(w_lab_i.dtype)

        active_body = counts > 0
        clump_body = active_body & (counts > 1)

        # representative (per-body) inertia and quaternion (average via weights)
        I_body = jax.ops.segment_sum(I_i * (movable * w)[:, None], ID_i, num_segments=N)      # (N,ang_dim)
        q_w_body = jax.ops.segment_sum(qw_i * (movable * w)[:, None], ID_i, num_segments=N)   # (N,1)
        q_xyz_body = jax.ops.segment_sum(qxyz_i * (movable * w)[:, None], ID_i, num_segments=N)  # (N,3)
        q_body = Quaternion.unit(Quaternion(q_w_body, q_xyz_body))

        # representative angular velocity in lab
        w_body_lab = jax.ops.segment_sum(w_lab_i * (movable * w)[:, None], ID_i, num_segments=N)  # (N,ang_dim)

        if dim == 2:
            w_body_body = w_body_lab  # scalar about z
        else:
            w3 = w_body_lab
            w_body_body = Quaternion.rotate_back(q_body, w3)  # (N,3) in body frame

        # clumps only
        mask = clump_body.astype(w_body_body.dtype)[:, None]
        w_body_body = w_body_body * mask
        I_body = I_body * mask

        return 0.5 * jnp.sum(I_body * (w_body_body * w_body_body))

    ke_flat = jax.vmap(_one)(ID, inertia, fixed, angVel, qw, qxyz)
    return ke_flat.reshape(lead_shape) if lead_shape else ke_flat.reshape(())


def assign_random_velocities_temperature(state, key, temperature, *, kB=1.0):
    """
    Assign random translational velocities for all bodies and random angular velocities for clump bodies,
    remove center-of-mass drift (per sample), and rescale to match the requested temperature.

    - Works for snapshots and for any leading dims; if a batch dim exists it must be axis 0.
    - `temperature` may be a scalar (shared across batch) or an array of shape (B,).
    - Uses equipartition with kB (default 1.0):
        E_trans_target = (dof_trans/2) kB T,  E_rot_target = (dof_rot/2) kB T
      and rescales translation and rotation *separately*.
    """
    N, dim = state.N, state.dim
    ang_dim = 1 if dim == 2 else 3

    lead_shape = state.ID.shape[:-1]
    M = int(jnp.prod(jnp.array(lead_shape))) if lead_shape else 1

    T_full = _as_batched_temperatures(state, temperature)
    T_flat = T_full.reshape((M,)) if lead_shape else jnp.asarray(T_full).reshape((1,))

    ID = state.ID.reshape((M, N))
    mass = state.mass.reshape((M, N))
    inertia = state.inertia.reshape((M, N, ang_dim))
    fixed = state.fixed.reshape((M, N))
    qw = state.q.w.reshape((M, N, 1))
    qxyz = state.q.xyz.reshape((M, N, 3))

    def _one(k, T_i, ID_i, m_i, I_i, fixed_i, qw_i, qxyz_i):
        counts, w = _body_weights_from_ids(ID_i, N)
        active_body = counts > 0
        clump_body = active_body & (counts > 1)

        movable = (1.0 - fixed_i).astype(float)

        # per-body mass (count each body once)
        m_body = jax.ops.segment_sum(m_i * movable * w, ID_i, num_segments=N)  # (N,)
        body_movable = m_body > 0

        # representative quaternion/inertia per body
        q_body = Quaternion.unit(
            Quaternion(
                jax.ops.segment_sum(qw_i * (movable * w)[:, None], ID_i, num_segments=N),
                jax.ops.segment_sum(qxyz_i * (movable * w)[:, None], ID_i, num_segments=N),
            )
        )
        I_body = jax.ops.segment_sum(I_i * (movable * w)[:, None], ID_i, num_segments=N)  # (N,ang_dim)

        k_v, k_w = jax.random.split(k, 2)

        # --- translation: sample per body, remove COM drift (movable bodies), rescale ---
        v_by_id = jax.random.normal(k_v, (N, dim))

        # remove COM drift using movable bodies only
        P = jnp.sum((m_body[:, None] * v_by_id) * body_movable.astype(v_by_id.dtype)[:, None], axis=0)
        Mtot = jnp.sum(m_body * body_movable.astype(m_body.dtype))
        v_com = jnp.where(Mtot > 0, P / Mtot, jnp.zeros((dim,), dtype=v_by_id.dtype))
        v_by_id = v_by_id - v_com[None, :]
        v_by_id = jnp.where(body_movable[:, None], v_by_id, 0.0)

        dof_t = dim * jnp.sum((active_body & body_movable).astype(float))
        E_t = 0.5 * jnp.sum(m_body * jnp.sum(v_by_id * v_by_id, axis=-1))
        E_t_target = 0.5 * kB * T_i * dof_t
        s_t = jnp.where((E_t > 0) & (dof_t > 0), jnp.sqrt(E_t_target / E_t), 0.0)
        v_by_id = v_by_id * s_t

        vel_new = v_by_id[ID_i]
        vel_new = vel_new * movable[:, None]

        # --- rotation: clumps only; sample omega in body frame, rotate to lab, rescale ---
        if dim == 2:
            w_body = jax.random.normal(k_w, (N, 1))
            w_lab = w_body
        else:
            w_body = jax.random.normal(k_w, (N, 3))
            w_lab = Quaternion.rotate(q_body, w_body)

        # mask out spheres + fixed bodies
        rot_movable = (clump_body & body_movable).astype(w_lab.dtype)[:, None]
        w_lab = w_lab * rot_movable

        # compute KE in body frame
        if dim == 2:
            w_body_eff = w_lab
        else:
            w_body_eff = Quaternion.rotate_back(q_body, w_lab)

        dof_r = ang_dim * jnp.sum((clump_body & body_movable).astype(float))
        E_r = 0.5 * jnp.sum(I_body * (w_body_eff * w_body_eff))
        E_r_target = 0.5 * kB * T_i * dof_r
        s_r = jnp.where((E_r > 0) & (dof_r > 0), jnp.sqrt(E_r_target / E_r), 0.0)
        w_lab = w_lab * s_r

        ang_new = w_lab[ID_i]
        ang_new = ang_new * movable[:, None]

        return vel_new, ang_new

    keys = jax.random.split(key, M)
    vel_flat, ang_flat = jax.vmap(_one)(
        keys, T_flat, ID, mass, inertia, fixed, qw, qxyz
    )

    state.vel = vel_flat.reshape(lead_shape + (N, dim)) if lead_shape else vel_flat.reshape((N, dim))
    state.angVel = ang_flat.reshape(lead_shape + (N, ang_dim)) if lead_shape else ang_flat.reshape((N, ang_dim))
    return state

def _dof_total(state):
    """Total DOF per sample for equipartition (translation for all bodies + rotation for clump bodies)."""
    N, dim = state.N, state.dim
    ang_dim = 1 if dim == 2 else 3

    counts = jnp.bincount(state.ID, length=N)          # (N,)
    active_body = counts > 0
    clump_body = counts > 1

    w = 1.0 / counts[state.ID]                         # (N,)
    m = (1.0 - state.fixed).astype(float)              # (N,)

    # movable per body (True if any member is movable; via weighted sum > 0)
    movable_body = jax.ops.segment_sum(m * w, state.ID, num_segments=N) > 0  # (N,)

    dof_t = dim * jnp.sum((active_body & movable_body).astype(float))
    dof_r = ang_dim * jnp.sum((clump_body & movable_body).astype(float))
    return dof_t + dof_r


def energies_one(state, system, *, kB=1.0):
    # PE: sum per-particle energies
    pe = jnp.sum(system.collider.compute_potential_energy(state, system))

    ke_t = kinetic_energy_translation(state)
    ke_r = kinetic_energy_rotation(state)

    dof = _dof_total(state)
    Tinst = jnp.where(dof > 0, 2.0 * (ke_t + ke_r) / (kB * dof), 0.0)

    # return scalars
    return pe, ke_t, ke_r, Tinst


jax.config.update("jax_enable_x64", True)

data_root = '/home/mmccraw/dev/data/01-01-26/specific-heat'

def count_dynamic_dofs(state: jd.State, subtract_drift: bool, is_rigid: bool) -> Tuple[int, int, int]:
    cids, offsets = jnp.unique(state.ID, return_index=True)
    free_mask = 1 - state.fixed[offsets]
    free_count = jnp.sum(free_mask)
    n_dof_v = (free_count - subtract_drift) * state.vel.shape[1]
    n_dof_w = free_count * state.angVel.shape[1] * is_rigid
    n_dof = n_dof_v + n_dof_w
    return n_dof, n_dof_v, n_dof_w

def _assign_random_velocities(state: jd.State, subtract_drift: bool, seed: Optional[float] = None) -> jd.State:
    if seed is None:
        seed = np.random.randint(0, 1e9)
    key = jax.random.PRNGKey(seed)
    v_k, w_k = jax.random.split(key, 2)
    cids, offsets = jnp.unique(state.ID, return_index=True)
    free_mask = 1 - state.fixed[offsets]
    v_clump = jax.random.normal(v_k, (cids.size, state.dim)) * free_mask[:, None]
    v_clump -= jnp.mean(v_clump, axis=0) * subtract_drift
    state.vel = v_clump[state.ID]
    w_clump = jax.random.normal(w_k, (cids.size, state.angVel.shape[1])) * free_mask[:, None]  # body frame
    w = w_clump[state.ID]
    if state.dim == 2:
        state.angVel = w
    else:  # rotate to lab frame
        state.angVel = state.q.rotate(state.q, w)
    return state

def calculate_translational_kinetic_energy(state: jd.State) -> jnp.array:
    cids, offsets = jnp.unique(state.ID, return_index=True)
    return 0.5 * jnp.sum((((1 - state.fixed) * state.mass)[:, None] * (state.vel ** 2))[offsets], axis=-1)

def calculate_rotational_kinetic_energy(state: jd.State) -> jnp.array:
    cids, offsets = jnp.unique(state.ID, return_index=True)
    w_body = state.q.rotate_back(state.q, state.angVel)  # to body frame
    return 0.5 * jnp.sum((((1 - state.fixed)[:, None] * state.inertia) * (w_body ** 2))[offsets], axis=-1)

def calculate_temperature(state: jd.State, is_rigid: bool, subtract_drift: bool, k_B: Optional[float] = 1.0) -> float:
    n_dof, _, _ = count_dynamic_dofs(state, subtract_drift, is_rigid)
    ke_t = calculate_translational_kinetic_energy(state)
    if is_rigid:
        ke_r = calculate_rotational_kinetic_energy(state)
    else:
        ke_r = 0.0
    ke = jnp.sum(ke_t + ke_r, axis=-1)
    return 2 * ke / (k_B * n_dof)

def set_temperature(state: jd.State, target_temperature: float, is_rigid: bool, subtract_drift: bool, seed: Optional[int] = None, k_B: Optional[float] = 1.0) -> jd.State:
    # assign random
    state = _assign_random_velocities(state, subtract_drift, seed)
    # count dofs
    n_dof, n_dof_v, n_dof_w = count_dynamic_dofs(state, subtract_drift, is_rigid)
    # impose equipartition
    state.vel *= n_dof_v / n_dof
    state.angVel *= n_dof_w / n_dof
    # calculate temperature
    temperature = calculate_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = target_temperature / temperature
    state.vel *= scale
    state.angVel *= scale  # do i need to rotate here?
    return state

def scale_to_temperature(state: jd.State, target_temperature: float, is_rigid: bool, subtract_drift: bool, k_B: Optional[float] = 1.0) -> jd.State:
    # subtract drift
    state.vel -= jnp.mean(state.vel, axis=0) * subtract_drift
    # calculate temperature
    temperature = calculate_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = target_temperature / temperature
    state.vel *= scale
    state.angVel *= scale  # do i need to rotate here?
    return state

if __name__ == "__main__":
    jamming_root = os.path.join(data_root, "jamming")
    for particle_type in os.listdir(jamming_root):
        root = os.path.join(jamming_root, particle_type)
        state = jd.utils.h5.load(os.path.join(root, "state.h5"))
        old_system = jd.utils.h5.load(os.path.join(root, "system.h5"))

        system = jd.System.create(
            state_shape=state.shape,
            dt=old_system.dt,
            linear_integrator_type="verlet",
            rotation_integrator_type="verletspiral",
            domain_type="periodic",
            force_model_type="spring",
            collider_type="naive",
            mat_table=old_system.mat_table,
            domain_kw=dict(
                box_size=old_system.domain.box_size,
            ),
        )

        n_steps = 1_0
        save_stride = 100
        n_snapshots = n_steps // save_stride


        target_temperature = 1e-5
        subtract_drift = True
        is_rigid = True
        # DO IS_DEFORMABLE TOO!!!!!!!!!!!
        # ^ AS PART OF THIS, ALSO DO A COM VELOCITY CALCULATOR
        k_B = 1.0
        seed = np.random.randint(0, 1e9)
        state = set_temperature(state, target_temperature, is_rigid, subtract_drift, seed, k_B)

        print(calculate_temperature(state, is_rigid, subtract_drift, k_B))

        # state, system, (state_traj, system_traj) = system.trajectory_rollout(
        #     state, system, n=n_snapshots, stride=save_stride
        # )

        # print(state.vel)

        exit()