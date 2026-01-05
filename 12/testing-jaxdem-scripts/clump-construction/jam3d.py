from typing import Sequence, Tuple
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

def generate_asperities_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    aspect_ratio: Sequence[float] = [1.0, 1.0, 1.0],
    add_core: bool = False,
    use_uniform_mesh: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    ____
    returns:
    asperity_positions: jnp.ndarray - (num_vertices + add_core, 3) array of positions of the asperities
    asperity_radii: jnp.ndarray - (num_vertices + add_core,) array of radii of the asperities
    ____
    notes:
    creates a particle composed of a set of surface asperities
    places asperities along either a sphere or an ellipsoid in 3d
    ensures that the outer-most length of the particle is equal to 2 * particle_radius
    adds a core which is useful for covering up large gaps between adjacent asperities
    the number of subdivisions for the icosphere mesh is suggested from target_num_vertices
    """
    if len(aspect_ratio) != 3:
        raise ValueError(f'Error: aspect ratio must be a 3-length list-like.  Expected 3, got {len(aspect_ratio)}')
    aspect_ratio = np.array(aspect_ratio)
    aspect_ratio /= np.min(aspect_ratio)
    if asperity_radius > particle_radius:
        print(f'Warning: asperity radius exceeds particle radius.  {asperity_radius} > {particle_radius}')
    core_radius = particle_radius - asperity_radius
    m = trimesh.creation.icosphere(subdivisions=num_trimesh_subdivisions(target_num_vertices), radius=core_radius)
    m.apply_scale(aspect_ratio)
    if use_uniform_mesh and np.sum(aspect_ratio) > 3:
        # when using an ellipsoid, re-mesh to ensure the vertices are evenly spaced
        # this avoids asperities bunching up at the major axes
        raise ValueError('Using uniform mesh isnt supported yet')
    asperity_positions = m.vertices
    asperity_radii = np.ones(m.vertices.shape[0]) * asperity_radius
    if add_core:
        if np.all(aspect_ratio == 1.0):
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

def make_single_particle_3d(
    asperity_radius: float,
    particle_radius: float,
    target_num_vertices: int,
    aspect_ratio: Sequence[float] = jnp.ones(3),
    add_core: bool = True,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(3),
    mass: float = 1.0,
    mesh_subdivisions: int = 4
    ) -> jd.State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: Sequence[float] - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    mesh_subdivisions: int - optional number of subdivisions when making the icosphere mesh to define the mass
    ____
    returns:
    single_clump_state: State - jaxdem state object containing the single clump particle in 3d
    """
    asperity_positions, asperity_radii = generate_asperities_3d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=target_num_vertices,
        aspect_ratio=aspect_ratio,
        add_core=add_core,
        use_uniform_mesh=use_uniform_mesh
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

def generate_ga_clump_system(particle_radii, sphere_pos, **kwargs):
    radii, counts = np.unique(particle_radii, return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    merged_state = None
    pbar = tqdm(particle_radii, total=len(particle_radii), desc='Generating clumps')
    for i, radius in enumerate(radii):
        state = make_single_particle_3d(
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
state = generate_ga_clump_system(particle_radii, sphere_pos, asperity_radius=asperity_radius, target_num_vertices=nv)

e_int = 1.0
dt = 1e-2

mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

system = jd.System.create(
    state_shape=state.shape,
    dt=dt,
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

state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-2)

jd.utils.h5.save(state, 'jammed_state_3d.h5')
jd.utils.h5.save(system, 'jammed_system_3d.h5')

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
    "jammed_3d.png",
    "1000",
], check=True)
