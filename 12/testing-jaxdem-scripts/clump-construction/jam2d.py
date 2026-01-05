from typing import Sequence, Tuple

import jax.numpy as jnp
import jax

import jaxdem as jd

from tqdm import tqdm

from shapely.geometry import Point
from shapely import affinity
from shapely.ops import unary_union

jax.config.update("jax_enable_x64", True)

def jax_copy(x):
    return jax.tree.map(lambda y: y, x)

def generate_asperities_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    add_core: bool = False,
    use_uniform_mesh: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipse)
    num_vertices: int - number of asperities
    aspect_ratio: float - optional aspect ratio of the ellipse
    add_core: bool - whether to construct the particles with a solid core
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipses
    ____
    returns:
    asperity_positions: jnp.ndarray - (num_vertices + add_core, 2) array of positions of the asperities
    asperity_radii: jnp.ndarray - (num_vertices + add_core,) array of radii of the asperities
    ____
    notes:
    creates a particle composed of a set of surface asperities
    places asperities along either a circle or an ellipse in 2d
    ensures that the outer-most length of the particle is equal to 2 * particle_radius
    adds a core which is useful for covering up large gaps between adjacent asperities
    """
    core_radius = particle_radius - asperity_radius
    if asperity_radius > particle_radius:
        print(f'Warning: asperity radius exceeds particle radius.  {asperity_radius} > {particle_radius}')
    if aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
    a = core_radius
    b = core_radius / aspect_ratio
    circle = Point(0.0, 0.0).buffer(1.0, quad_segs=1000 * num_vertices)
    if use_uniform_mesh and aspect_ratio != 1.0:
        # when making an ellipse, select the points evenly along the outer perimeter
        # this avoids asperities bunching up at the major axis
        ellipse = affinity.scale(
            circle,
            xfact=a,
            yfact=b
        )
        # distances = jnp.sort(jnp.random.uniform(0, ellipse.length, num_vertices))  # for random case
        distances = jnp.arange(num_vertices) * ellipse.length / num_vertices
        points = [ellipse.boundary.interpolate(d) for d in distances]
        asperity_positions = jnp.array([[p.x, p.y] for p in points])
    else:
        distances = jnp.arange(num_vertices) * circle.length / num_vertices
        points = [circle.boundary.interpolate(d) for d in distances]
        asperity_positions = jnp.array([[p.x * a, p.y * b] for p in points])
    asperity_radii = jnp.ones(num_vertices) * asperity_radius

    if add_core:
        if aspect_ratio == 1.0:
            asperity_positions = jnp.concatenate((asperity_positions, jnp.zeros((1, 2))), axis=0)
            asperity_radii = jnp.concatenate((asperity_radii, jnp.array([core_radius])), axis=0)
        else:
            print('Warning: ellipse core not yet supported')
    return asperity_positions, asperity_radii

def make_single_particle_2d(
    asperity_radius: float,
    particle_radius: float,
    num_vertices: int,
    aspect_ratio: float = 1.0,
    add_core: bool = True,
    use_uniform_mesh: bool = False,
    particle_center: Sequence[float] = jnp.zeros(2),
    mass: float = 1.0,
    quad_segs: int = 10_000
    ) -> jd.State:
    """
    asperity_radius: float - radius of the asperities
    particle_radius: float - outer-most radius of the particle (major axis if an ellipsoid)
    target_num_vertices: int - target number of asperities - usually not met due to icosphere subdivision
    aspect_ratio: float - optional aspect ratios of the ellipsoid
    add_core: bool - whether to construct the particles with a solid core
    use_uniform_mesh: bool - whether to use uniformly spaced vertices, only relevant for ellipsoids
    particle_center: Sequence[float] - optional particle center location
    mass: float - optional mass of the entire particle
    quad_segs: int - optional number of segments used to define the mass
    ____
    returns:
    single_clump_state: State - jaxdem state object containing the single clump particle in 2d
    """

    asperity_positions, asperity_radii = generate_asperities_2d(
        asperity_radius=asperity_radius,
        particle_radius=particle_radius,
        num_vertices=num_vertices,
        aspect_ratio=aspect_ratio,
        add_core=add_core,
        use_uniform_mesh=use_uniform_mesh
    )

    shape = unary_union([
        Point(p).buffer(r, quad_segs=quad_segs) for p, r in zip(asperity_positions, asperity_radii)
    ])

    single_clump_state = jd.State.create(
        pos=asperity_positions + particle_center,
        rad=asperity_radii,
        ID=jnp.zeros(asperity_positions.shape[0]),
        volume=jnp.ones(asperity_positions.shape[0]) * shape.area / asperity_positions.shape[0]
    )

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=0.5)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    single_clump_state = jd.utils.compute_clump_properties(single_clump_state, mat_table, n_samples=50_000)

    true_mass = jnp.ones_like(single_clump_state.mass) * mass
    single_clump_state.inertia *= (true_mass / single_clump_state.mass)[..., None]
    single_clump_state.mass = true_mass

    return single_clump_state


def generate_ga_clump_system(particle_radii, sphere_pos, vertex_counts, **kwargs):
    dim = sphere_pos.shape[1]
    radii, index, counts = jnp.unique(particle_radii, return_index=True, return_counts=True)
    offsets = jnp.concatenate((jnp.zeros(1), jnp.cumsum(counts))).astype(int)
    merged_state = None
    pbar = tqdm(particle_radii, total=len(particle_radii), desc='Generating clumps')
    for i, (radius, j) in enumerate(zip(radii, index)):
        nv = vertex_counts[j]
        if dim == 2:
            state = make_single_particle_2d(
                particle_radius=radius,
                num_vertices=nv,
                **kwargs
            )
        elif dim == 3:
            state = make_single_particle_2d(
                particle_radius=radius,
                num_vertices=nv,
                **kwargs
            )
        else:
            raise ValueError(f'dim: {dim} not supported')
        for j in range(offsets[i], offsets[i + 1]):
            new_state = jax_copy(state)
            new_state.pos_c = jnp.ones_like(new_state.pos_c) * sphere_pos[j]
            if j == 0:
                merged_state = new_state
            else:
                merged_state = jd.State.merge(merged_state, new_state)
            pbar.update(1)
    return merged_state


import numpy as np

N = 50
phi = 0.4
dim = 2
asperity_radius = 0.1

particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
nv = np.ones_like(particle_radii).astype(int)
nv[particle_radii == np.min(particle_radii)] *= 10
nv[particle_radii == np.max(particle_radii)] *= 14
sphere_pos, box_size = jd.utils.random_sphere_configuration(particle_radii, phi, dim)
state = generate_ga_clump_system(
    particle_radii,
    sphere_pos,
    asperity_radius=asperity_radius,
    vertex_counts=nv,
    aspect_ratio=5.0,
    use_uniform_mesh=True,
    add_core=False
)


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

jd.utils.h5.save(state, 'jammed_state_2d.h5')
jd.utils.h5.save(system, 'jammed_system_2d.h5')

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
    "jammed_2d.png",
    "1000",
], check=True)


# make the particles of different sizes have different numbers of vertices in nthe sysytem creator

# give the particles random orientations

# make script to create any initial system of ga particles in 2d/3d
# make script to jam any initial system of ga particles in 2d/3d