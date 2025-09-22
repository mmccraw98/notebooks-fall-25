import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff, calc_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import NeighborMethod, DT_INT, DT_FLOAT
from pydpmd.plot import draw_particles_frame
import subprocess
import os
import shutil
from tqdm import tqdm


def create_rigid_bumpy_system_from_n_vertices(n_vertices_per_particle: np.ndarray, radii: np.ndarray, packing_fraction: float, mu_eff: float, which: str):
    """
    All particles have the same vertex radii
    """
    if radii.shape != n_vertices_per_particle.shape:
        raise ValueError(f'size mismatch: radii.shape: {radii.shape} n_vertices_per_particle.shape: {n_vertices_per_particle.shape}')
    if which not in ['avg', 'small']:
        raise ValueError(f'argument which: "{which}" not understood, must be one of "avg", "small"')
    
    # define units
    particle_radius = 0.5
    particle_mass = 1.0
    e_interaction = 1.0
    if which == 'avg':  # set the average radii to particle_radius if not already
        radii = radii / np.mean(radii) * particle_radius
    elif which == 'small':  # set the small radii to particle_radius, preserve relative scales
        radii = radii / np.min(radii) * particle_radius
    else:
        raise NotImplementedError(f'which: "{which}" not handled')

    # create system object
    rb = RigidBumpy()
    rb.allocate_particles(n_vertices_per_particle.size)
    rb.allocate_systems(1)
    rb.allocate_vertices(n_vertices_per_particle.sum())
    rb.n_vertices_per_particle = n_vertices_per_particle.astype(DT_INT)
    rb.set_ids()
    rb.validate()
    rb.rad = radii.astype(DT_FLOAT)
    rb.mass.fill(particle_mass)
    rb.e_interaction.fill(e_interaction)
    rb.calculate_uniform_vertex_mass()
    
    # assign friction coefficients

    # handle single vertex particles
    multi_vertex_mask = n_vertices_per_particle > 1
    if np.any(~multi_vertex_mask):
        rb.vertex_rad[~multi_vertex_mask[rb.vertex_particle_id]] = rb.rad[~multi_vertex_mask]
    if np.any(multi_vertex_mask):
        if which == 'avg':
            i = np.argmin(np.abs(radii[multi_vertex_mask] - np.mean(radii[multi_vertex_mask])))
            print(get_closest_vertex_radius_for_mu_eff(mu_eff, rb.rad[i], rb.n_vertices_per_particle[i]))
        elif which == 'small':
            pass
        else:
            raise NotImplementedError(f'which: "{which}" not handled')


def generate_bidisperse_radii(total: int, count_ratio: float, size_ratio: float, small_size: float = 0.5):
    """
    total: number of particles
    count_ratio: proportion of particles that are the small size
    size_ratio: large diameter in terms of the small diameter
    small_size: diameter of the small particle
    """
    if count_ratio > 1 or count_ratio < 0:
        raise ValueError('count ratio out of bounds: 0 <= count_ratio <= 1')
    if size_ratio < 0:
        raise ValueError('size ratio out of bounds: 0 <= size_ratio')
    n_small = int(total * count_ratio)
    n_large = total - n_small
    radii = np.ones(total).astype(DT_FLOAT)
    radii[:n_small] = small_size
    radii[n_small:] = small_size * size_ratio
    return radii

def generate_polydisperse_radii(total: int, std_dev: float, avg_size: float = 0.5, random_seed: int = 42):
    """
    total: number of particles
    std_dev: standard deviation of the gaussian
    avg_size: size of the average particle
    """
    print("TODO: verify that this is the right way to handle polydispersity")
    np.random.seed(random_seed)
    return np.random.normal(size=total, loc=avg_size, scale=std_dev)

def build_rigid_bumpy_system_from_radii(radii: np.ndarray, which: str, mu_eff: float, nv: int, packing_fraction: float, add_core: bool = False):
    """
    radii: np.ndarray containing the radius for each particle
    which: str defining what the target particle is - one of "avg" (design around the average particle) or "small" (design around the small particle)
    mu_eff: float friction coefficient of the target particle.  other particles are attempted to be built with this friction coefficient
    nv: int number of vertices in the target particle
    packing_fraction: float packing fraction of the final system
    add_core: bool whether or not to build the particles with cores
    """
    min_nv = 2
    max_nv = 1000
    nv_trial = np.arange(min_nv, max_nv)

    # define units
    particle_radius = 0.5
    particle_mass = 1.0
    e_interaction = 1.0

    if which == 'avg':  # set the average radii to particle_radius if not already
        radii = radii / np.mean(radii) * particle_radius
    elif which == 'small':  # set the small radii to particle_radius, preserve relative scales
        radii = radii / np.min(radii) * particle_radius
    else:
        raise NotImplementedError(f'which: "{which}" not handled')

    # get the vertex radius for the target particle (given by which) (now corresponds to radius = particle_radius) and its nv
    target_particle_id = np.argmin(np.abs(radii - particle_radius))
    vertex_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, radii[target_particle_id], nv)
    # solve for n_vertices given mu_eff, particle radius, and vertex radius
    n_vertices_per_particle = np.ones_like(radii).astype(DT_INT)
    count = 0
    for rad in np.unique(radii):
        mask = radii == rad
        diffs = np.abs(calc_mu_eff(vertex_radius, rad, nv_trial) - mu_eff)
        nv_pred = nv_trial[~np.isnan(diffs)][np.argmin(diffs[~np.isnan(diffs)])]
        n_vertices_per_particle[mask] = nv_pred
    if np.any(np.diff(np.unique(n_vertices_per_particle)) > int(0.6 * np.min(n_vertices_per_particle[n_vertices_per_particle > 1]))):
        print('WARNING: large variation detected in n_vertices_per_particle!  unique values:', np.unique(n_vertices_per_particle))
    
    # create system object
    rb = RigidBumpy()
    rb.using_core = add_core
    if add_core:
        n_vertices_per_particle[n_vertices_per_particle > 1] += 1
    rb.allocate_particles(n_vertices_per_particle.size)
    rb.allocate_systems(1)
    rb.allocate_vertices(n_vertices_per_particle.sum())
    rb.n_vertices_per_particle = n_vertices_per_particle.astype(DT_INT)
    rb.set_ids()
    rb.validate()
    rb.rad = radii.astype(DT_FLOAT)
    rb.vertex_rad.fill(vertex_radius)
    multi_vertex_mask = np.where(rb.n_vertices_per_particle > 1)[0]
    if np.any(multi_vertex_mask) and add_core:  # assign the core radius as the inner radius (outer radius - vertex radius)
        rb.vertex_rad[rb.particle_offset[1:][multi_vertex_mask] - 1] = rb.rad[multi_vertex_mask] - rb.vertex_rad[rb.particle_offset[:-1][multi_vertex_mask]]

    # handle the single vertex particles, set their vertex radii to be equal to their particle radii
    single_vertex_mask = rb.n_vertices_per_particle == 1
    rb.vertex_rad[single_vertex_mask[rb.vertex_particle_id]] = rb.rad[single_vertex_mask]

    # set the masses and inertia
    rb.mass.fill(particle_mass)
    rb.e_interaction.fill(e_interaction)
    rb.calculate_uniform_vertex_mass()

    # set the positions in the box and scale to a packing fraction
    rb.box_size.fill(1.0)
    rb.set_positions(0, 0)
    rb.set_vertices_on_particles_as_disk()
    
    rb.calculate_inertia()
    rb.scale_to_packing_fraction(packing_fraction)

    return rb

if __name__ == "__main__":
    # ensure this can handle single vertex case?
    radii = generate_polydisperse_radii(10, 1e-1)  # use 'avg'
    which = 'avg'
    # radii = generate_bidisperse_radii(100, 0.5, 1.4)  # use 'small'
    # which = 'small'

    mu_eff = 1
    nv = 3
    packing_fraction = 0.5
    add_core = True

    rb = build_rigid_bumpy_system_from_radii(
        radii, which, mu_eff, nv, packing_fraction, add_core
    )

    print(np.unique(rb.n_vertices_per_particle))

    # issues:
    # using_core needs to be saved properly
    # any undefined / misc arrays in the particle class should not be saved
    # TODO: pydpmd: ensure that extraneous / undefined arrays are not saved to the h5!

    # TODO: dpmd: when saving to init in append mode, only save a field if it doesnt already exist in the h5


    rb.calculate_mu_eff()

    print(rb.mu_eff)

    import matplotlib.pyplot as plt
    draw_particles_frame(0, plt.gca(), rb, system_id=0, which='vertex', cmap_name='viridis', location=None)
    plt.savefig('test.png')