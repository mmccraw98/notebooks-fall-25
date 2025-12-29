from tqdm import tqdm
import trimesh
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation
from jaxdem.utils import Quaternion
import jax.numpy as jnp
import jaxdem as jd

def jax_copy(x):
    """
    Make a *real* copy of a pytree containing JAX arrays.

    This matters because some jitted JAXDEM kernels donate input buffers for speed.
    If we call such a function repeatedly while reusing the same `state`/`system`
    objects (or views into their buffers), JAX will error with:
      "Invalid buffer passed: buffer has been deleted or donated."
    """
    def _copy_leaf(y):
        # Copy only array-like leaves; leave Python objects as-is.
        if isinstance(y, (jax.Array, jnp.ndarray, np.ndarray)):
            # `jax.lax.copy` isn't a public API in many JAX versions; `.copy()`/`jnp.copy`
            # forces a new device buffer.
            try:
                return y.copy()
            except Exception:
                return jnp.copy(jnp.asarray(y))
        return y
    return jax.tree.map(_copy_leaf, x)

def num_trimesh_vertices(subdivisions):
    # count the number of vertices for a set number of subdivisions
    return 10 * 4 ** subdivisions + 2

def num_trimesh_subdivisions(num_vertices):
    # count the number of subdisions to get a set number of vertices
    s = round(np.log10((num_vertices - 2) / 10) / np.log10(4))
    return max(s, 0)  # clip to 0

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

def generate_mesh(asperity_positions, asperity_radii, subdivisions):
    meshes = []
    for a, r in zip(asperity_positions, asperity_radii):
            m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)
            m.apply_translation(a)
            meshes.append(m)
    mesh = trimesh.util.concatenate(meshes)
    # assert (mesh.is_winding_consistent & mesh.is_watertight)
    assert mesh.is_volume
    return mesh

def make_single_particle(
    asperity_radius,
    particle_radius,
    nv,
    aspect_ratio,
    add_core,
    particle_center,
    mass,
    clump_id=0,
    mesh_subdivisions=4,
):
    asperity_positions, asperity_radii = generate_asperities(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=nv,
        aspect_ratio=aspect_ratio,
        add_core=add_core
    )
    # THIS IS SOMEHOW INCORRECT
    # mesh = generate_mesh(
    #     asperity_positions=asperity_positions,
    #     asperity_radii=asperity_radii,
    #     subdivisions=mesh_subdivisions
    # )
    single_clump_state = jd.State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        # IMPORTANT: each physical clump must have a distinct ID, otherwise "unique IDs"
        # and any clump-level logic (centers, separation, etc.) will be wrong.
        ID=jnp.ones(asperity_positions.shape[0], dtype=jnp.int32) * int(clump_id),
        # volume=jnp.ones(asperity_positions.shape[0]) * mesh.volume
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=0.5)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = jd.utils.compute_clump_properties(single_clump_state, mat_table, n_samples=50_000)

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state

def compress_free_particle(state, increment):
    direction = jnp.diff(state.pos_c[indices], axis=0)
    direction /= jnp.linalg.norm(direction)
    state.pos_c += increment * direction * (1 - state.fixed)[:, None]
    return state

mass = 1.0
asperity_radius = 0.4
particle_radius = 0.5
nv = 5
aspect_ratio = np.array([1.0, 1.0, 1.0])
add_core = True


dim = 3
phi = 0.4
radii = [0.5, 0.7]
nvs = [10, 40]
mesh_subdivisions = 5

# THERE IS AN ERROR IN THE CALCULATION OF THE CLUMP VOLUME FROM THE MESH SOMEHOW

state = None
for i, (radius, nv) in enumerate(zip(radii, nvs)):
    new_state = make_single_particle(
        asperity_radius=asperity_radius,
        particle_radius=radius,
        nv=nv,
        aspect_ratio=aspect_ratio,
        add_core=add_core,
        particle_center=jnp.zeros(dim),
        mass=mass,
        clump_id=i,
        mesh_subdivisions=mesh_subdivisions
    )
    if i == 0:
        new_state.fixed = jnp.ones(new_state.pos.shape[:-1], dtype=bool)
        state = new_state
    else:
        offset_pos = np.array(new_state.pos_c)
        offset_pos[:, 0] += sum(radii)
        new_state.pos_c = jnp.array(offset_pos)
        state = jd.State.merge(state, new_state)


e_int = 1.0
dt = 1e-2
mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
system = jd.System.create(
    state_shape=state.shape,
    dt=dt,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
    domain_type="free",
    force_model_type="spring",
    collider_type="naive",
    # collider_type="celllist",
    # collider_kw=dict(state=state),
    mat_table=mat_table,
)



# find the minimum radius for the current configuration
#
# We want the *minimum* center-to-center separation at which the clumps are just
# non-overlapping. For purely repulsive contacts, total contact potential energy
# is monotonic in separation (decreases to 0). That makes it a good bisection target.
#
# Approach:
# - Build a bracket [low, high] such that E(low) > 0 (overlap) and E(high) == 0 (no overlap)
# - Bisection on E(mid) to find the boundary where E -> 0+
#
_, indices = jnp.unique(state.ID, return_index=True)
if int(indices.shape[0]) != 2:
    raise ValueError(
        f"Expected exactly 2 clumps (2 unique IDs), got {int(indices.shape[0])}. "
        "Check that you set distinct IDs when building the state."
    )

base_state = jax_copy(state)
base_system = jax_copy(system)

def _center_sep_and_dir(s):
    centers = s.pos_c[indices]  # (2, 3)
    v = centers[1] - centers[0]
    dist = jnp.linalg.norm(v)
    direction = v / (dist + 1e-30)
    return dist, direction

base_rad, base_dir = _center_sep_and_dir(base_state)
move_mask = (1 - base_state.fixed)[:, None]

def total_contact_energy_at_rad(target_rad):
    # Reset to baseline each evaluation so we don't accumulate numerical drift.
    s = jax_copy(base_state)
    cur_rad, cur_dir = _center_sep_and_dir(s)
    # Use the baseline direction to keep the motion 1D along the initial separation axis.
    # (If you want to allow sliding/rotation, that's a different minimization problem.)
    delta = (target_rad - cur_rad) * base_dir
    s.pos_c = s.pos_c + delta * move_mask
    # IMPORTANT: pass fresh copies each call so donated buffers don't get reused.
    sys_in = jax_copy(base_system)
    s, sys_ = sys_in.collider.compute_force(s, sys_in)
    e = jnp.sum(sys_.collider.compute_potential_energy(s, sys_))
    return float(np.asarray(e))

rad_step0 = 1e-2
max_bracket_iter = 60
max_bisect_iter = 80
rad_rtol = 1e-10
energy_tol = 1e-14

e0 = total_contact_energy_at_rad(base_rad)
print(f"Initial separation rad={float(np.asarray(base_rad)):.8g}, E={e0:.8g}")

# Bracket the transition (overlap -> non-overlap).
if e0 > energy_tol:
    # Currently overlapping: base_rad is a LOWER bound.
    rad_low = float(np.asarray(base_rad))
    rad_high = rad_low
    step = rad_step0
    for _ in range(max_bracket_iter):
        rad_high = rad_high + step
        eh = total_contact_energy_at_rad(rad_high)
        if eh <= energy_tol:
            break
        step *= 2.0
    else:
        raise RuntimeError("Failed to find a non-overlapping upper bracket (E==0) by expanding rad_high.")
else:
    # Currently non-overlapping: base_rad is an UPPER bound.
    rad_high = float(np.asarray(base_rad))
    rad_low = rad_high
    step = rad_step0
    for _ in range(max_bracket_iter):
        rad_low = max(rad_low - step, 0.0)
        el = total_contact_energy_at_rad(rad_low)
        if el > energy_tol:
            break
        step *= 2.0
        if rad_low == 0.0:
            break
    if total_contact_energy_at_rad(rad_low) <= energy_tol:
        raise RuntimeError("Failed to find an overlapping lower bracket (E>0) by shrinking rad_low.")

print(f"Bracket: rad_low={rad_low:.8g} (E>0), rad_high={rad_high:.8g} (E≈0)")

# Bisection.
for i in range(1, max_bisect_iter + 1):
    mid = 0.5 * (rad_low + rad_high)
    em = total_contact_energy_at_rad(mid)
    if em > energy_tol:
        rad_low = mid
    else:
        rad_high = mid

    if (rad_high - rad_low) / max(rad_high, 1e-30) < rad_rtol:
        break

rad_star = rad_high  # smallest separation with ~zero contact energy
print(f"Done: rad*={rad_star:.12g} after {i} bisection iterations")
print(f"Check: E(rad*)={total_contact_energy_at_rad(rad_star):.8g}")



last_pos = state.pos_c.copy()

