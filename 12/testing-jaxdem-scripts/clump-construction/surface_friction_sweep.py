"""
Surface friction sweep (geometric friction from asperity-clump contacts).

This script builds two clumps (a fixed "base" particle and a moving "tracer"),
places the tracer at a surface location on the base specified by spherical angles
(theta, phi), and then compresses it along the local surface normal to match a
prescribed normal load (Fn_target) using a 1D bisection on center-to-center separation.

The "effective friction coefficient" is reported as:

    mu_eff = |F_t| / max(|F_n|, eps)

where F is the total contact force on the tracer clump, decomposed into normal and
tangential components at the current surface point.

Important note:
  This measures *geometric friction* from roughness (asperities) using the normal-only
  spring force law currently implemented in JaxDEM (`forces/spring.py`). If you want
  Coulomb/Mindlin friction, we can add a tangential contact law, but that requires
  contact-history state (tangential spring) that JaxDEM doesn't currently store.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import trimesh

import jaxdem as jd
from jaxdem.utils import Quaternion

jax.config.update("jax_enable_x64", True)

def num_trimesh_subdivisions(num_vertices: int) -> int:
    s = round(np.log10((num_vertices - 2) / 10) / np.log10(4))
    return max(s, 0)


def generate_asperities(asperity_radius, particle_radius, target_num_vertices, aspect_ratio=[1.0, 1.0, 1.0], add_core=False):
    # builds the locations of all the asperities on the surface of an ellipsoidal particle
    # the asperities will all have uniform radius and will decorate the surface of an icosphere mesh
    # the icosphere mesh will be initially generated for a sphere with a set number of subdivisions
    # the number of subdivisions is suggested from the desired number of vertices
    # the icosphere mesh is then scaled by the aspect ratio to give an ellipsoid
    if len(aspect_ratio) != 3:
        raise ValueError(f'Error: aspect ratio must be a 3-length list-like.  Expected 3, got {len(aspect_ratio)}')
    aspect_ratio = np.array(aspect_ratio)
    if asperity_radius > particle_radius:
        print(f'Warning: asperity radius exceeds particle radius.  {asperity_radius} > {particle_radius}')
    core_radius = particle_radius - asperity_radius
    m = trimesh.creation.icosphere(subdivisions=num_trimesh_subdivisions(target_num_vertices), radius=core_radius)
    m.apply_scale(aspect_ratio)
    asperity_positions = m.vertices
    asperity_radii = np.ones(m.vertices.shape[0]) * asperity_radius
    if add_core:
        if np.all(aspect_ratio == 1.0):  # sphere branch
            asperity_positions = np.concatenate((asperity_positions, np.zeros((1, 3))), axis=0)
            asperity_radii = np.concatenate((asperity_radii, np.array([core_radius])), axis=0)
        else:
            print('Warning: ellipsoid core not yet supported')
    return asperity_positions, asperity_radii


def jax_copy(x):
    def _copy_leaf(y):
        if isinstance(y, (jax.Array, jnp.ndarray, np.ndarray)):
            try:
                return y.copy()
            except Exception:
                return jnp.copy(jnp.asarray(y))
        return y

    return jax.tree.map(_copy_leaf, x)


def euler_xyz_to_quaternion(euler_xyz: jnp.ndarray) -> Quaternion:
    # Rotation.as_quat() returns (x,y,z,w). JaxDEM Quaternion stores (w, xyz).
    q_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return Quaternion.create(w=q_xyzw[..., 3:4], xyz=q_xyzw[..., 0:3])


def spherical_unit(theta: float, phi: float) -> jnp.ndarray:
    st = jnp.sin(theta)
    return jnp.array([st * jnp.cos(phi), st * jnp.sin(phi), jnp.cos(theta)])


def spherical_basis(theta: float, phi: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (e_r, e_theta, e_phi) in the *same frame* the angles are defined in.
    """
    st, ct = jnp.sin(theta), jnp.cos(theta)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    e_r = jnp.array([st * cp, st * sp, ct])
    e_theta = jnp.array([ct * cp, ct * sp, -st])  # increasing theta
    e_phi = jnp.array([-sp, cp, 0.0])  # increasing phi
    return e_r, e_theta, e_phi


def set_clump_pose(state: jd.State, clump_id: int, pos_c: jnp.ndarray, q: Quaternion) -> jd.State:
    mask = state.ID == clump_id
    mask_v = mask[..., None]

    state.pos_c = jnp.where(mask_v, pos_c, state.pos_c)

    w_full = jnp.broadcast_to(q.w, state.q.w.shape)
    xyz_full = jnp.broadcast_to(q.xyz, state.q.xyz.shape)
    state.q = Quaternion(
        w=jnp.where(mask_v, w_full, state.q.w),
        xyz=jnp.where(mask_v, xyz_full, state.q.xyz),
    )
    return state


def translate_clump(state: jd.State, clump_id: int, delta: jnp.ndarray) -> jd.State:
    mask = (state.ID == clump_id)[..., None]
    state.pos_c = state.pos_c + mask * delta
    return state


def clump_force(state: jd.State, clump_id: int) -> jnp.ndarray:
    # Every particle in the clump has the same total force after collider segment_sum,
    # so just pick the first index.
    idx = jnp.argmax((state.ID == clump_id).astype(jnp.int32))
    return state.force[idx]


@dataclass
class SweepConfig:
    asperity_radius: float = 0.2
    add_core: bool = True
    base_radius: float = 0.5
    tracer_radius: float = 0.5
    base_nv: int = 160
    tracer_nv: int = 160
    mesh_aspect_ratio: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    e_int: float = 1.0
    dt: float = 1e-2
    rad_rtol: float = 1e-10
    max_bracket_iter: int = 80
    max_bisect_iter: int = 80
    fn_target: float = 1e-2


def make_single_clump(
    cfg: SweepConfig,
    *,
    particle_radius: float,
    nv: int,
    clump_id: int,
    particle_center: jnp.ndarray,
    mass: float = 1.0,
):
    asperity_positions, asperity_radii = generate_asperities(
        asperity_radius=cfg.asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=nv,
        aspect_ratio=cfg.mesh_aspect_ratio,
        add_core=cfg.add_core,
    )

    st = jd.State.create(
        pos=jnp.asarray(asperity_positions, dtype=float) + particle_center,
        rad=jnp.asarray(asperity_radii, dtype=float),
        ID=jnp.ones((asperity_positions.shape[0],), dtype=jnp.int32) * int(clump_id),
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=0.5)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    st = jd.utils.compute_clump_properties(st, mat_table, n_samples=50_000)

    true_mass = jnp.ones_like(st.mass) * mass
    st.inertia *= (true_mass / st.mass)[..., None]
    st.mass = true_mass
    return st


def normalize_clump_reference_frame(state: jd.State, clump_id: int) -> jd.State:
    """
    Set the clump's reference orientation to identity by absorbing the current
    quaternion into `pos_p`. After this, `q=identity` reproduces the same geometry.
    """
    mask = state.ID == clump_id
    mask_v = mask[..., None]

    # Extract a representative quaternion (all particles in the clump should match).
    idx = jnp.argmax(mask.astype(jnp.int32))
    q0 = Quaternion(w=state.q.w[idx : idx + 1], xyz=state.q.xyz[idx : idx + 1])

    # Rotate body-frame offsets into lab, then set q to identity.
    pos_p_lab = q0.rotate(q0, state.pos_p)
    state.pos_p = jnp.where(mask_v, pos_p_lab, state.pos_p)

    qI = Quaternion.create(w=jnp.ones((1, 1)), xyz=jnp.zeros((1, 3)))
    w_full = jnp.broadcast_to(qI.w, state.q.w.shape)
    xyz_full = jnp.broadcast_to(qI.xyz, state.q.xyz.shape)
    state.q = Quaternion(
        w=jnp.where(mask_v, w_full, state.q.w),
        xyz=jnp.where(mask_v, xyz_full, state.q.xyz),
    )
    return state


def total_contact_energy(state: jd.State, system: jd.System) -> float:
    e = jnp.sum(system.collider.compute_potential_energy(state, system))
    return float(np.asarray(e))


def compute_clump_force(state: jd.State, system: jd.System, clump_id: int) -> jnp.ndarray:
    """
    Compute the net (segment-summed) contact force on a clump.

    Note: collider kernels donate buffers, so we always pass fresh copies.
    """
    st = jax_copy(state)
    sys_ = jax_copy(system)
    st, sys_ = sys_.collider.compute_force(st, sys_)
    return clump_force(st, clump_id)


@partial(jax.jit, static_argnames=("src_id", "dst_id"))
def net_force_between_ids(
    state: jd.State, system: jd.System, *, src_id: int, dst_id: int
) -> jnp.ndarray:
    """
    Net force on `src_id` clump due to pair interactions with `dst_id` clump,
    computed explicitly (O(N^2)) without calling collider kernels.

    This is useful inside inner loops (e.g., bisection) because collider kernels
    are usually jitted with buffer donation.
    """
    N = state.N
    iota = jax.lax.iota(dtype=int, size=N)
    jota = jax.lax.iota(dtype=int, size=N)

    def force_row(i):
        forces_ij, _ = jax.vmap(system.force_model.force, in_axes=(None, 0, None, None))(
            i, jota, state, system
        )
        mask = (state.ID[i] == src_id) & (state.ID[jota] == dst_id)
        return (forces_ij * mask[:, None]).sum(axis=0)

    return jax.vmap(force_row)(iota).sum(axis=0)


@partial(
    jax.jit,
    static_argnames=("base_id", "tracer_id", "max_bracket_iter", "max_bisect_iter"),
)
def find_separation_for_normal_force(
    base_state: jd.State,
    system: jd.System,
    *,
    base_id: int,
    tracer_id: int,
    n_hat: jnp.ndarray,
    rad0: float,
    fn_target: float,
    rad_rtol: float,
    max_bracket_iter: int,
    max_bisect_iter: int,
) -> float:
    """
    1D bracket + bisection on normal force to find rad such that Fn(rad) ~= fn_target.

    Assumes Fn increases as rad decreases (more compression).
    """
    base_com = base_state.pos_c[jnp.argmax((base_state.ID == base_id).astype(jnp.int32))]
    n_hat = n_hat / (jnp.linalg.norm(n_hat) + 1e-30)

    def fn_at(rad: jax.Array) -> jax.Array:
        pos_c_new = jnp.where(
            (base_state.ID == tracer_id)[..., None],
            base_com + rad * n_hat,
            base_state.pos_c,
        )
        st = replace(base_state, pos_c=pos_c_new)
        F = net_force_between_ids(st, system, src_id=tracer_id, dst_id=base_id)
        return jnp.dot(F, n_hat)

    # Assume rad0 is sufficiently large (Fn(rad0) <= fn_target). We only compress to bracket.
    rad_high = jnp.asarray(rad0, dtype=float)
    rad_low = jnp.asarray(rad0, dtype=float)
    step0 = jnp.asarray(1e-3, dtype=float)

    def bracket_cond(carry):
        rad_low, step, k = carry
        return jnp.logical_and(fn_at(rad_low) < fn_target, k < max_bracket_iter)

    def bracket_body(carry):
        rad_low, step, k = carry
        rad_low = jnp.maximum(rad_low - step, 0.0)
        step = step * 2.0
        return rad_low, step, k + 1

    rad_low, _, _ = jax.lax.while_loop(bracket_cond, bracket_body, (rad_low, step0, 0))

    def bisect_cond(carry):
        rad_low, rad_high, k = carry
        rel = (rad_high - rad_low) / jnp.maximum(rad_high, 1e-30)
        return jnp.logical_and(k < max_bisect_iter, rel > rad_rtol)

    def bisect_body(carry):
        rad_low, rad_high, k = carry
        mid = 0.5 * (rad_low + rad_high)
        rad_low, rad_high = jax.lax.cond(
            fn_at(mid) >= fn_target,
            lambda _: (mid, rad_high),
            lambda _: (rad_low, mid),
            operand=None,
        )
        return rad_low, rad_high, k + 1

    rad_low, _, _ = jax.lax.while_loop(bisect_cond, bisect_body, (rad_low, rad_high, 0))
    return rad_low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-theta", type=int, default=9)
    ap.add_argument("--n-phi", type=int, default=18)
    ap.add_argument("--theta-min", type=float, default=1e-3)
    ap.add_argument("--theta-max", type=float, default=np.pi - 1e-3)
    ap.add_argument("--out", type=str, default="surface_friction_sweep.npz")
    ap.add_argument("--fn-target", type=float, default=1e-2, help="Target normal load (force) applied to tracer along local surface normal.")
    ap.add_argument(
        "--base-euler",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Base clump Euler angles (xyz) in radians.",
    )
    ap.add_argument(
        "--tracer-euler",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Tracer clump Euler angles (xyz) in radians.",
    )
    ap.add_argument(
        "--tracer-euler-in-base-frame",
        action="store_true",
        help="Interpret tracer Euler angles in the base particle's body frame (q_tracer = q_base @ q_rel).",
    )
    args = ap.parse_args()

    cfg = SweepConfig(fn_target=args.fn_target)

    # Build two clumps with distinct IDs.
    base = make_single_clump(
        cfg,
        particle_radius=cfg.base_radius,
        nv=cfg.base_nv,
        clump_id=0,
        particle_center=jnp.zeros(3),
    )
    tracer = make_single_clump(
        cfg,
        particle_radius=cfg.tracer_radius,
        nv=cfg.tracer_nv,
        clump_id=1,
        particle_center=jnp.zeros(3),
    )
    base.fixed = jnp.ones(base.pos.shape[:-1], dtype=bool)
    print(base.N, tracer.N)

    # Offset tracer so we start well-separated.
    tracer = translate_clump(tracer, 1, jnp.array([cfg.base_radius + cfg.tracer_radius + 0.5, 0.0, 0.0]))

    state = jd.State.merge(base, tracer)
    # Normalize both clumps so user-specified Euler angles are applied in a stable reference frame.
    state = normalize_clump_reference_frame(state, 0)
    state = normalize_clump_reference_frame(state, 1)

    mats = [jd.Material.create("elastic", young=cfg.e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=cfg.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_type="free",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
    )

    # Set orientations.
    q_base = euler_xyz_to_quaternion(jnp.array(args.base_euler))
    q_tracer_rel = euler_xyz_to_quaternion(jnp.array(args.tracer_euler))
    q_tracer = q_tracer_rel if not args.tracer_euler_in_base_frame else (q_base @ q_tracer_rel)

    base_com = state.pos_c[jnp.argmax((state.ID == 0).astype(jnp.int32))]
    tracer_com0 = state.pos_c[jnp.argmax((state.ID == 1).astype(jnp.int32))]
    state = set_clump_pose(state, 0, pos_c=base_com, q=q_base)
    state = set_clump_pose(state, 1, pos_c=tracer_com0, q=q_tracer)

    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, args.n_phi, endpoint=False)

    mu_out = np.zeros((args.n_theta, args.n_phi), dtype=float)
    fn_out = np.zeros_like(mu_out)
    ft_out = np.zeros_like(mu_out)
    rad_out = np.zeros_like(mu_out)

    # Initial guess: sum of radii.
    rad_guess = float(cfg.base_radius + cfg.tracer_radius)

    for it, theta in enumerate(thetas):
        for ip, phi in enumerate(phis):
            # Define the surface direction in *base body frame*, then rotate to lab.
            e_r_b, e_th_b, e_ph_b = spherical_basis(theta, phi)
            n_hat = q_base.rotate(q_base, e_r_b)
            # We'll compute the *magnitude* of the tangential component (direction-free):
            # Ft_vec = F - (F·n)n

            # Place tracer along this normal and find separation giving the target normal load.
            rad_star = find_separation_for_normal_force(
                state,
                system,
                base_id=0,
                tracer_id=1,
                n_hat=n_hat,
                rad0=rad_guess,
                fn_target=cfg.fn_target,
                rad_rtol=cfg.rad_rtol,
                max_bracket_iter=cfg.max_bracket_iter,
                max_bisect_iter=cfg.max_bisect_iter,
            )
            rad_guess = rad_star

            st = jax_copy(state)
            st = set_clump_pose(st, 1, pos_c=base_com + rad_star * n_hat, q=q_tracer)

            F = net_force_between_ids(st, system, src_id=1, dst_id=0)
            Fn = jnp.dot(F, n_hat)
            Ft_vec = F - Fn * n_hat
            Ft = jnp.linalg.norm(Ft_vec)
            mu = Ft / jnp.maximum(jnp.abs(Fn), 1e-30)

            mu_out[it, ip] = float(np.asarray(mu))
            fn_out[it, ip] = float(np.asarray(Fn))
            ft_out[it, ip] = float(np.asarray(Ft))
            rad_out[it, ip] = float(rad_star)

    np.savez(
        args.out,
        thetas=thetas,
        phis=phis,
        mu=mu_out,
        Fn=fn_out,
        Ft=ft_out,
        rad=rad_out,
        base_euler=np.asarray(args.base_euler),
        tracer_euler=np.asarray(args.tracer_euler),
        tracer_euler_in_base_frame=np.asarray(bool(args.tracer_euler_in_base_frame)),
        cfg=cfg.__dict__,
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()


