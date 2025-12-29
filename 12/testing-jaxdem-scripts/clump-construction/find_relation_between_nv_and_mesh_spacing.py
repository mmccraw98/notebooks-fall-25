import trimesh
import numpy as np
import jax
from jax.scipy.spatial.transform import Rotation
from jaxdem.utils import Quaternion
import jax.numpy as jnp
import jaxdem as jd
def num_trimesh_vertices(subdivisions):
    # count the number of vertices for a set number of subdivisions
    return 10 * 4 ** subdivisions + 2

def num_trimesh_subdivisions(num_vertices):
    # count the number of subdisions to get a set number of vertices
    s = round(np.log10((num_vertices - 2) / 10) / np.log10(4))
    return max(s, 0)  # clip to 0

def generate_asperity_mesh(asperity_radius, particle_radius, target_num_vertices, aspect_ratio=[1.0, 1.0, 1.0]):
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
    return m

add_core = False
particle_center = np.zeros(3)

particle_radius = 0.5
aspect_ratio = np.array([1.0, 1.0, 1.0])

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

nv_range = 10 * np.arange(1, 20)
asperity_radius_range = np.linspace(0.01, particle_radius, 10)

cmap = plt.cm.viridis
norm = Normalize(vmin=min(asperity_radius_range), vmax=max(asperity_radius_range))

for asperity_radius in asperity_radius_range:
    mean_edge_length = []
    for nv in nv_range:

        mesh = generate_asperity_mesh(
            asperity_radius=asperity_radius,
            particle_radius=particle_radius,
            target_num_vertices=nv,
            aspect_ratio=aspect_ratio,
        )

        # faces = mesh.vertices[mesh.faces]  # maybe useful too
        edges = mesh.vertices[mesh.edges]  # n_edges, n_vertices, n_dim
        edge_lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=-1)  # n_edges
        # edge_lengths, counts = np.unique(edge_lengths, return_counts=True)
        mean_edge_length.append(np.mean(edge_lengths))

    plt.plot(nv_range, mean_edge_length / asperity_radius, c=cmap(norm(asperity_radius)))
plt.yscale('log')
plt.savefig('edge_lengths.png')
plt.close()