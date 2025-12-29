import trimesh
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation
from jaxdem.utils import Quaternion
import jax.numpy as jnp
import jaxdem as jd

def jax_copy(x):
    return jax.tree.map(lambda y: y, x)

def make_grid_state(n_per_axis, dim):
    radius = 0.5
    spacing = 2 * radius
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
    assert (mesh.is_winding_consistent & mesh.is_watertight)
    return mesh

def make_single_particle(asperity_radius, particle_radius, nv, aspect_ratio, add_core, particle_center, mass, mesh_subdivisions=4):
    asperity_positions, asperity_radii = generate_asperities(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=nv,
        aspect_ratio=aspect_ratio,
        add_core=add_core
    )
    mesh = generate_mesh(
        asperity_positions=asperity_positions,
        asperity_radii=asperity_radii,
        subdivisions=mesh_subdivisions
    )
    single_clump_state = jd.State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0]) * mesh.volume
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=0.5)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = jd.utils.compute_clump_properties(single_clump_state, mat_table, n_samples=50_000)

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state

mass = 1.0
asperity_radius = 0.2
particle_radius = 0.5
nv = 20
aspect_ratio = np.array([1.0, 1.0, 1.0])
add_core = True
particle_center = np.zeros(3)


N = 9
dim = 3
phi = 0.1
count_ratios = [0.1, 0.4, 0.5]
size_ratios = [1.0, 1.4, 1.2]
small_radius = 0.5
mesh_subdivisions = 4

count_ratios = np.asarray(count_ratios)
size_ratios = np.asarray(size_ratios)
assert len(count_ratios) == len(size_ratios)
count_ratios /= sum(count_ratios)
size_ratios /= min(size_ratios)

counts = np.round(N * count_ratios).astype(int)
sizes = small_radius * size_ratios
# print(sum(counts), N)  # must be equal
# print(counts / N == count_ratios) # warn if not match
nvs = np.ones_like(counts) * nv  # TODO: make this such that the friction is constant across all shapes

assert np.all(sizes > 0)
assert np.all(counts > 0)

count_offset = np.concatenate(([0], np.cumsum(counts)))

# make set grid position
# TODO: replace with random energy minimized positions of spheres / disks according to phi
grid_state, box_size = make_grid_state(int(N ** (1 / dim)), dim)
grid_pos = grid_state.pos_c
particle_radii = np.concatenate([np.ones(c) * s for c, s in zip(counts, sizes)])

particle_volume = jnp.sum(jnp.exp(0.5 * dim * jnp.log(jnp.pi) + dim * jnp.log(particle_radii) - jax.scipy.special.gammaln(0.5 * dim + 1.0)))
scale = (particle_volume / phi) ** (1 / dim) / np.mean(box_size)
box_size *= scale
grid_pos *= scale
assert np.isclose(phi, particle_volume / np.prod(box_size))

final_state = None
for i, (count, size, nv) in enumerate(zip(counts, sizes, nvs)):
    new_state = make_single_particle(
        asperity_radius=asperity_radius,
        particle_radius=size,
        nv=nv,
        aspect_ratio=aspect_ratio,
        add_core=add_core,
        particle_center=particle_center,
        mass=mass,
        mesh_subdivisions=mesh_subdivisions
    )

    for j in range(count_offset[i], count_offset[i + 1]):
        state = jax_copy(new_state)
        state.pos_c = jnp.ones_like(state.pos_c) * grid_pos[j]
        if j == 0:
            final_state = state
        else:
            final_state = jd.State.merge(final_state, state)

state = final_state

_, indices = jnp.unique(state.ID, return_index=True)
print(jnp.sum(state.volume[indices]) / jnp.prod(box_size))



e_int = 1.0
dt = 1e-2

mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

system = jd.System.create(
    state_shape=state.shape,
    dt=dt,
    # linear_integrator_type="verlet",
    # rotation_integrator_type="verletspiral",
    linear_integrator_type="linearfire",
    rotation_integrator_type="rotationfire",
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    # collider_type="celllist",
    # collider_kw=dict(state=state),
    mat_table=mat_table,
    domain_kw=dict(
        box_size=box_size,
    ),
)

# n_steps = 1_000_0
# save_stride = 100
# n_snapshots = n_steps // save_stride

# seed = np.random.randint(0, 1000000)
# key = jax.random.PRNGKey(seed)
# key_vel, key_angVel = jax.random.split(key, 2)
# cid, offsets = jnp.unique(state.ID, return_index=True)
# N_clumps = cid.size
# clump_vel = jax.random.normal(key_vel, (N_clumps, state.dim))
# clump_vel -= jnp.mean(clump_vel, axis=0)
# state.vel = clump_vel[state.ID]
# ke_t = jnp.sum(0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1))
# # ke_r ?????
# dof = (state.dim + state.inertia.shape[-1]) * N_clumps - state.dim
# current_temp = 2 * ke_t / dof
# scale = jnp.sqrt(1e-5 / current_temp)
# state.vel *= scale

# state, syste, (state_traj, system_traj) = system.trajectory_rollout(
#     state, system, n=n_snapshots, stride=save_stride
# )
# _, indices = jnp.unique(state.ID, return_index=True)
# Ke_t = 0.5 * jnp.sum(
#     (state_traj.mass * jnp.vecdot(state_traj.vel, state_traj.vel))[:, indices],
#     axis=1,
# )
# angVel = state_traj.q.rotate_back(state_traj.q, state_traj.angVel)
# Ke_r = 0.5 * jnp.sum(
#     jnp.vecdot(angVel, state_traj.inertia * angVel)[:, indices], axis=1
# )
# Pe = jnp.sum(
#     jax.vmap(system_traj.collider.compute_potential_energy)(
#         state_traj, system_traj
#     ),
#     axis=1,
# )
# E = Ke_t + Ke_r + Pe

# import matplotlib.pyplot as plt
# plt.plot(E, label="Energy")
# plt.plot(Pe, label="Pe")
# plt.plot(Ke_t, label="Ke_t")
# plt.plot(Ke_r, label="Ke_r")
# plt.legend()
# plt.savefig(f"energies.png", dpi=400)
# plt.close()

# import subprocess
# from pathlib import Path
# import h5py
# with h5py.File("traj.h5", "w") as f:
#     f.create_dataset("pos", data=np.asarray(state_traj.pos))
#     f.create_dataset("rad", data=np.asarray(state_traj.rad))
#     f.create_dataset("ID", data=np.asarray(state_traj.ID))
#     f.create_dataset("box_size", data=np.asarray(system_traj.domain.box_size))

# # --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
# script_dir = Path(__file__).resolve().parent
# run_animation = script_dir.parent / "animation" / "run_animation.sh"
# subprocess.run(
#     [
#         str(run_animation),
#         "traj.h5",
#         "traj.gif",
#         "100",   # num_frames (evenly sampled if traj has more)
#         "1000",  # base_pixels
#         "5",    # fps
#     ],
#     check=True,
# )



state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-4)

import subprocess
from pathlib import Path
import h5py
with h5py.File('config.h5', 'w') as f:
    f.create_dataset("pos", data=np.asarray(state.pos))
    f.create_dataset("rad", data=np.asarray(state.rad))
    f.create_dataset("ID",  data=np.asarray(state.ID))
    f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
script_dir = Path(__file__).resolve().parent
run_render = script_dir.parent / "rigid-particle-creation" / "run_render.sh"
subprocess.run([
    str(run_render),
    "config.h5",
    "jammed.png",
    "1000",
], check=True)