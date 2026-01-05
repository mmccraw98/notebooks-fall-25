from tqdm import tqdm
import trimesh
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation
from jaxdem.utils import Quaternion
import jax.numpy as jnp
import jaxdem as jd

jax.config.update("jax_enable_x64", True)

def jax_copy(x):
    return jax.tree.map(lambda y: y, x)

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
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(r))
        m.apply_translation(a)
        meshes.append(m)
    engines = getattr(trimesh.boolean, "engines_available", set())
    if "manifold" in engines:
        mesh = trimesh.boolean.union(meshes, engine="manifold")
    elif None in engines:
        mesh = trimesh.boolean.union(meshes, engine=None)
    else:
        raise RuntimeError(
            "No trimesh boolean backend is available; can't union sphere meshes. "
            "Install one (recommended: `pip install manifold3d`)."
        )

    assert mesh.is_volume
    return mesh

def make_single_particle(asperity_radius, particle_radius, nv, aspect_ratio=np.ones(3), add_core=True, particle_center=np.zeros(3), mass=1.0, mesh_subdivisions=4):
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
        volume=jnp.ones(asperity_positions.shape[0]) * mesh.volume / asperity_positions.shape[0]
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=0.5)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = jd.utils.compute_clump_properties(single_clump_state, mat_table, n_samples=50_000)

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state

def generate_clump_system(particle_radii, sphere_pos, **kwargs):
    radii, counts = np.unique(particle_radii, return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    merged_state = None
    pbar = tqdm(particle_radii, total=len(particle_radii), desc='Generating clumps')
    for i, radius in enumerate(radii):
        state = make_single_particle(
            particle_radius=radius,
            **kwargs
        )
        for j in range(offsets[i], offsets[i + 1]):
            new_state = jax_copy(state)
            new_state.pos_c = jnp.ones_like(new_state.pos_c) * sphere_pos[j]
            if j == 0:
                merged_state = new_state
            else:
                merged_state = jd.State.merge(merged_state, new_state)
            pbar.update(1)
    return merged_state

N = 50
phi = 0.4
dim = 3
asperity_radius = 0.3
nv = 5

particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
sphere_pos, box_size = jd.utils.random_sphere_configuration(particle_radii, phi, dim)
state = generate_clump_system(particle_radii, sphere_pos, asperity_radius=asperity_radius, nv=nv)

e_int = 1.0
dt = 1e-2

mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

system = jd.System.create(
    state_shape=state.shape,
    dt=1e-2,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
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
scale = jnp.sqrt(1e-4 / current_temp)
state.vel *= scale

n_steps = 1_000_0
save_stride = 100
n_snapshots = n_steps // save_stride
final_state, final_system, (traj_state, traj_system) = jd.System.trajectory_rollout(
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
plt.savefig('energies.png')
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
        "traj.gif",
        "100",   # num_frames (evenly sampled if traj has more)
        "1000",  # base_pixels
        "15",    # fps
    ],
    check=True,
)