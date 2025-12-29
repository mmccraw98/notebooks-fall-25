import trimesh
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation
from jaxdem.utils import Quaternion
import jax.numpy as jnp

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

def random_unit_quat_3d(key):
    u = jax.random.normal(key, (4,))
    u = u / jnp.linalg.norm(u)
    return Quaternion(w=u[0:1], xyz=u[1:4])

def random_rotate(q, key):
    R = random_unit_quat_3d(key)
    Rb = Quaternion(
        w=jnp.broadcast_to(R.w, q.w.shape),
        xyz=jnp.broadcast_to(R.xyz, q.xyz.shape),
    )
    return Quaternion.unit(Rb @ q)

mass = 1.0
asperity_radius = 0.2
particle_radius = 0.5
nv = 20
aspect_ratio = [1.0, 1.0, 1.0]
add_core = True
mesh_subdivisions = 5

asperity_positions, asperity_radii = generate_asperities(
    asperity_radius=asperity_radius,
    particle_radius=particle_radius,
    target_num_vertices=nv,
    aspect_ratio=aspect_ratio,
    add_core=add_core
)
mesh = generate_mesh(asperity_positions, asperity_radii, mesh_subdivisions)

pos_c = mesh.mass_properties.center_mass
volume = mesh.volume
inertia = 0.5 * (mass / mesh.volume) * (mesh.mass_properties.inertia + mesh.mass_properties.inertia.T)
nv = asperity_positions.shape[0]

# get the body axis frame
vals, vecs = np.linalg.eigh(inertia)
rot = Rotation.from_matrix(vecs)
q_xyzw = rot.as_quat()
q_update = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])
q_update = jnp.stack([q_update for i in range(nv)])
q = Quaternion(q_update[..., 0:1], q_update[..., 1:])

# save in state
mass = jnp.ones(nv) * mass
volume = jnp.ones(nv) * volume
pos_c = jnp.stack([pos_c for i in range(nv)])
inertia = jnp.stack([vals for i in range(nv)])
pos_p = q.rotate_back(q, asperity_positions - pos_c)

# # randomly rotate
# seed = np.random.randint(0, 1e9)
# key = jax.random.PRNGKey(seed)
# q = random_rotate(q, key)
# pos_p = q.rotate_back(q, asperity_positions - pos_c)


import jaxdem as jd

v = np.random.normal(loc=0, scale=1e-2, size=(3))
vel = np.stack([v for i in range(nv)])
state1 = jd.State(
    pos_c=pos_c + 1,
    pos_p=pos_p,
    vel=vel,
    force=jnp.zeros_like(pos_c),
    q=q,
    angVel=jnp.zeros((nv, 3)),
    torque=jnp.zeros((nv, 3)),
    rad=jnp.array(asperity_radii),
    volume=volume,
    mass=mass,
    inertia=inertia,
    ID=jnp.zeros((nv,), dtype=int),
    mat_id=jnp.zeros((nv,), dtype=int),
    species_id=jnp.zeros((nv,), dtype=int),
    fixed=jnp.zeros((nv,), dtype=bool),
)
assert state1.is_valid

v = np.random.normal(loc=0, scale=1e-2, size=(3))
vel = np.stack([v for i in range(nv)])
state2 = jd.State(
    pos_c=pos_c + 2,
    pos_p=pos_p,
    vel=vel,
    force=jnp.zeros_like(pos_c),
    q=q,
    angVel=jnp.zeros((nv, 3)),
    torque=jnp.zeros((nv, 3)),
    rad=jnp.array(asperity_radii),
    volume=volume,
    mass=mass,
    inertia=inertia,
    ID=jnp.zeros((nv,), dtype=int) + 1,
    mat_id=jnp.zeros((nv,), dtype=int),
    species_id=jnp.zeros((nv,), dtype=int),
    fixed=jnp.zeros((nv,), dtype=bool),
)
assert state2.is_valid

v = np.random.normal(loc=0, scale=1e-2, size=(3))
vel = np.stack([v for i in range(nv)])
state3 = jd.State(
    pos_c=pos_c + 3,
    pos_p=pos_p,
    vel=vel,
    force=jnp.zeros_like(pos_c),
    q=q,
    angVel=jnp.zeros((nv, 3)),
    torque=jnp.zeros((nv, 3)),
    rad=jnp.array(asperity_radii),
    volume=volume,
    mass=mass,
    inertia=inertia,
    ID=jnp.zeros((nv,), dtype=int) + 2,
    mat_id=jnp.zeros((nv,), dtype=int),
    species_id=jnp.zeros((nv,), dtype=int),
    fixed=jnp.zeros((nv,), dtype=bool),
)
assert state3.is_valid

state = jd.State.merge(jd.State.merge(state1, state2), state3)

e_int = 1.0
dt = 1e-2
box_size = jnp.ones(3) * 4.0

mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

system = jd.System.create(
    state_shape=state.shape,
    dt=dt,
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

n_steps = 1_000_000
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

exit()

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