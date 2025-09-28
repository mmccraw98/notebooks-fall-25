import numpy as np
from pydpmd.data import RigidBumpy, load, BaseParticle, Disk, BasePointParticle, BasePolyParticle
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff, calc_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import NeighborMethod, DT_INT, DT_FLOAT
from pydpmd.plot import draw_particles_frame
import subprocess
import os
import shutil
from tqdm import tqdm

import time

import tempfile

import matplotlib.pyplot as plt


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

def build_disk_system_from_radii(radii: np.ndarray, which: str, packing_fraction: float, rng_seed: int = 0):
    np.random.seed(rng_seed)
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
    disk = Disk()
    disk.allocate_particles(radii.size)
    disk.allocate_systems(1)
    disk.allocate_vertices(0)
    disk.set_ids()
    disk.validate()
    disk.rad = radii.astype(DT_FLOAT)
    # set the masses and inertia
    disk.mass.fill(particle_mass)
    disk.e_interaction.fill(e_interaction)
    # set the positions in the box and scale to a packing fraction
    disk.box_size.fill(1.0)
    disk.set_positions(1, rng_seed)
    disk.scale_to_packing_fraction(packing_fraction)
    return disk

def build_rigid_bumpy_system_from_radii(radii: np.ndarray, which: str, mu_eff: float | list[float], nv: int | list[int], packing_fraction: float, add_core: bool | str = False, cap_nv: float = np.inf, n_duplicates: int = 1, draw_figures: bool = False):
    """
    radii: np.ndarray containing the radius for each particle
    which: str defining what the target particle is - one of "avg" (design around the average particle) or "small" (design around the small particle)
    mu_eff: float friction coefficient of the target particle.  other particles are attempted to be built with this friction coefficient
    nv: int number of vertices in the target particle
    packing_fraction: float __effective__ packing fraction of the final system
    add_core: bool whether or not to build the particles with cores or str indicating a special case to selectively add cores
    cap_nv: float target number of vertices * cap_nv gives the maximum number of vertices for any particle
    n_duplicates: int number of times to duplicate the system
    """
    if not isinstance(mu_eff, list):
        mu_eff = [mu_eff] * n_duplicates
    if not isinstance(nv, list):
        nv = [nv] * n_duplicates
    if len(mu_eff) != n_duplicates:
        raise ValueError(f"mu_eff must be a list of length {n_duplicates}")
    if len(nv) != n_duplicates:
        raise ValueError(f"nv must be a list of length {n_duplicates}")

    temp_path = tempfile.mkdtemp()
    # build the initial data and equilibrate it, ideally to a 0-overlap state
    disk = join_systems([build_disk_system_from_radii(radii, which, packing_fraction, int(np.random.randint(0, 1e9))) for _ in range(n_duplicates)])
    disk.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disk, 0.3)
    disk.save(temp_path)
    subprocess.run([
        os.path.join("/home/mmccraw/dev/dpmd/build/", "disk_equilibrate_pbc"),
        temp_path,
        temp_path,
    ], check=True)

    disk = load(temp_path, location=["final", "init"])
    shutil.rmtree(temp_path)
    rbs = []
    for i, disk in enumerate(split_systems(disk)):
        pos = disk.pos.copy()
        radii = disk.rad.copy()
        box_size = disk.box_size.copy()

        min_nv = 2 if mu_eff[i] > 0 else 1
        max_nv = nv[i] * cap_nv
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

        n_vertices_per_particle = np.ones_like(radii).astype(DT_INT)
        if mu_eff[i] > 0:
            target_particle_id = np.argmin(np.abs(radii - particle_radius))
            vertex_radius = get_closest_vertex_radius_for_mu_eff(mu_eff[i], radii[target_particle_id], nv[i])
            for rad in np.unique(radii):
                mask = radii == rad
                diffs = np.abs(calc_mu_eff(vertex_radius, rad, nv_trial) - mu_eff[i])
                n_vertices_per_particle[mask] = nv_trial[~np.isnan(diffs)][np.argmin(diffs[~np.isnan(diffs)])]

        # initial construction
        rb = RigidBumpy()
        if isinstance(add_core, str) and add_core == 'selective':
            _add_core = (mu_eff[i] >= 1) or (mu_eff[i] >= 0.5 and nv[i] >= 20)
        else:
            _add_core = add_core
        rb.using_core = _add_core and mu_eff[i] > 0
        if rb.using_core:
            n_vertices_per_particle[n_vertices_per_particle > 1] += 1
        rb.allocate_particles(n_vertices_per_particle.size)
        rb.allocate_systems(1)
        rb.allocate_vertices(n_vertices_per_particle.sum())
        rb.n_vertices_per_particle = n_vertices_per_particle.astype(DT_INT)
        rb.set_ids()
        rb.validate()

        # assign values from the equilibrated disks
        rb.box_size = box_size.astype(DT_FLOAT).copy()
        rb.set_positions(1, int(np.random.randint(0, 1e9)))  # basically done just to assign random angles
        rb.pos = pos.astype(DT_FLOAT).copy()  # overwrite the random positions with the disk positions
        rb.rad = radii.astype(DT_FLOAT).copy()
        rb.mass.fill(particle_mass)
        rb.e_interaction.fill(e_interaction)
        rb.calculate_uniform_vertex_mass()

        # assign vertex radii, handling all the edge cases (single vertex, multi-vertex with/without cores)
        single_vertex_mask = rb.n_vertices_per_particle == 1
        multi_vertex_mask = ~single_vertex_mask
        if np.any(single_vertex_mask):  # set the vertex radius to the particle radius for single vertex particles
            rb.vertex_rad[single_vertex_mask[rb.vertex_particle_id]] = rb.rad[single_vertex_mask]
        if np.any(multi_vertex_mask):
            rb.vertex_rad[multi_vertex_mask[rb.vertex_particle_id]] = vertex_radius  # every vertex has the same radius
            if rb.using_core:  # assign the core radius as the inner radius (outer radius - vertex radius)
                rb.vertex_rad[rb.particle_offset[1:][multi_vertex_mask] - 1] = rb.rad[multi_vertex_mask] - rb.vertex_rad[rb.particle_offset[:-1][multi_vertex_mask]]

        # assign the particle positions
        rb.set_vertices_on_particles_as_disk()
        rb.calculate_inertia()
        rb.calculate_packing_fraction()
        rbs.append(rb)
        if draw_figures:
            draw_particles_frame(None, plt.gca(), rb, system_id=0, which='vertex', cmap_name='viridis', location=None)
            plt.savefig(f'figures/test_{nv[i]}_{mu_eff[i]}.png')
            plt.close()

    return join_systems(rbs)

def set_standard_cell_list_parameters(particle: BaseParticle, alpha: float = 0.3):
    """
    Set the standard cell list parameters for a particle system.
    alpha: size of the verlet skin in terms of the largest particle diameter
    """
    if isinstance(particle, BasePolyParticle):
        max_diam = 2.0 * np.array([np.max(particle.vertex_rad[particle.vertex_system_offset[i]:particle.vertex_system_offset[i+1]]) for i in range(particle.n_systems())])
    elif isinstance(particle, BasePointParticle):
        max_diam = 2.0 * np.array([np.max(particle.rad[particle.system_offset[i]:particle.system_offset[i+1]]) for i in range(particle.n_systems())])
    else:
        raise NotImplementedError(f'type {type(particle)} not handled')
    guessed_cell_size = np.ones(particle.n_systems()).astype(DT_FLOAT) * (max_diam * (1 + alpha))
    particle.neighbor_cutoff = np.ones(particle.n_systems()).astype(DT_FLOAT) * (alpha * max_diam)
    particle.thresh2 = (np.ones_like(guessed_cell_size) * max_diam * alpha / 2.0) ** 2
    particle.cell_dim = np.floor(particle.box_size / guessed_cell_size[:, None]).astype(DT_INT)
    particle.cell_dim = np.clip(particle.cell_dim, 3, None)  # ensure that the cell dim is at least 3 because we use a 3-cell stencil
    particle.cell_size = np.ones_like(particle.box_size).astype(DT_FLOAT) * particle.box_size / particle.cell_dim
    particle.cell_system_start = np.concatenate(([0], np.cumsum(np.prod(particle.cell_dim, axis=1)))).astype(DT_INT)