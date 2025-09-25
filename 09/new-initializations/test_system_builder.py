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

def build_rigid_bumpy_system_from_radii(radii: np.ndarray, which: str, mu_eff: float, nv: int, packing_fraction: float, add_core: bool = False, rng_seed: int = 0, cap_nv: float = np.inf):
    """
    radii: np.ndarray containing the radius for each particle
    which: str defining what the target particle is - one of "avg" (design around the average particle) or "small" (design around the small particle)
    mu_eff: float friction coefficient of the target particle.  other particles are attempted to be built with this friction coefficient
    nv: int number of vertices in the target particle
    packing_fraction: float packing fraction of the final system
    add_core: bool whether or not to build the particles with cores
    rng_seed: int seed for the random number generator
    cap_nv: float target number of vertices * cap_nv gives the maximum number of vertices for any particle
    """
    np.random.seed(rng_seed)
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
    # solve for n_vertices given mu_eff, particle radius, and vertex radius
    n_vertices_per_particle = np.ones_like(radii).astype(DT_INT)
    if mu_eff > 0:
        vertex_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, radii[target_particle_id], nv)
        for rad in np.unique(radii):
            mask = radii == rad
            diffs = np.abs(calc_mu_eff(vertex_radius, rad, nv_trial) - mu_eff)
            nv_pred = nv_trial[~np.isnan(diffs)][np.argmin(diffs[~np.isnan(diffs)])]
            if cap_nv and nv_pred > cap_nv * nv:
                nv_pred = int(cap_nv * nv)
            n_vertices_per_particle[mask] = nv_pred
        if np.any(np.diff(np.unique(n_vertices_per_particle)) > int(0.6 * np.min(n_vertices_per_particle[n_vertices_per_particle > 1]))):
            print('WARNING: large variation detected in n_vertices_per_particle!  unique values:', np.unique(n_vertices_per_particle))
    
    # create system object
    rb = RigidBumpy()
    rb.using_core = add_core
    if add_core and mu_eff > 0:
        n_vertices_per_particle[n_vertices_per_particle > 1] += 1
    rb.allocate_particles(n_vertices_per_particle.size)
    rb.allocate_systems(1)
    rb.allocate_vertices(n_vertices_per_particle.sum())
    rb.n_vertices_per_particle = n_vertices_per_particle.astype(DT_INT)
    rb.set_ids()
    rb.validate()
    rb.rad = radii.astype(DT_FLOAT)
    if mu_eff > 0:
        rb.vertex_rad.fill(vertex_radius)
    else:
        rb.vertex_rad = rb.rad.copy()
    multi_vertex_mask = np.where(rb.n_vertices_per_particle > 1)[0]
    if np.any(multi_vertex_mask) and add_core:  # assign the core radius as the inner radius (outer radius - vertex radius)
        # the core is the last vertex radius for each particle
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
    rb.set_positions(1, rng_seed)
    rb.set_vertices_on_particles_as_disk()
    
    rb.calculate_inertia()
    rb.scale_to_packing_fraction(packing_fraction)

    return rb

def set_standard_cell_list_parameters(rb: RigidBumpy, alpha: float = 0.3):
    """
    Set the standard cell list parameters for a RigidBumpy object.
    alpha: size of the verlet skin in terms of the largest vertex diameter
    """
    max_vertex_diam = 2.0 * np.array([np.max(rb.vertex_rad[rb.vertex_system_offset[i]:rb.vertex_system_offset[i+1]]) for i in range(rb.n_systems())])
    rb.neighbor_cutoff = np.ones(rb.n_systems()).astype(DT_FLOAT) * (max_vertex_diam * (1 + alpha))
    rb.thresh2 = (np.ones_like(rb.neighbor_cutoff) * max_vertex_diam * alpha / 2.0) ** 2
    rb.cell_dim = np.floor(rb.box_size / rb.neighbor_cutoff[:, None]).astype(DT_INT)
    rb.cell_dim = np.clip(rb.cell_dim, 1, None)  # ensure that the cell dim is at least 1
    rb.cell_size = np.ones_like(rb.box_size).astype(DT_FLOAT) * rb.box_size / rb.cell_dim
    rb.cell_system_start = np.concatenate(([0], np.cumsum(np.prod(rb.cell_dim, axis=1)))).astype(DT_INT)

def get_minimized_rigid_bumpy_system(radii, which, mu_eff, nv, packing_fraction, add_core, rng_seed, cap_nv, script_path = os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_equilibrate_pbc")):
    # make a system of single vertex particles and minimize its energy, then use its positions to build a system of multi-vertex particles
    disks = build_rigid_bumpy_system_from_radii(radii, which, 0, nv, packing_fraction, False, rng_seed)
    disks.set_neighbor_method(NeighborMethod.Cell)
    set_standard_cell_list_parameters(disks, 0.3)
    tmp_path = tempfile.mkdtemp()
    disks.save(tmp_path)
    subprocess.run([
        script_path,
        tmp_path,
        tmp_path,
    ], check=True)
    disks = load(tmp_path, location=["final", "init"])
    shutil.rmtree(tmp_path)

    rb = build_rigid_bumpy_system_from_radii(radii, which, mu_eff, nv, packing_fraction, add_core, rng_seed, cap_nv=cap_nv)
    rb.box_size = disks.box_size.copy()
    rb.pos = disks.pos.copy()
    rb.set_vertices_on_particles_as_disk()
    rb.calculate_mu_eff()
    return rb

if __name__ == "__main__":
    # ensure this can handle single vertex case?
    # radii = generate_polydisperse_radii(10, 1e-1)  # use 'avg'
    # which = 'avg'
    radii = generate_bidisperse_radii(100, 0.5, 1.4)  # use 'small'
    which = 'small'
    packing_fraction = 0.5

    # state points
    # nvs = [3, 6, 10, 20]
    nvs = [6]
    # mu_effs = [0.01, 0.05, 0.1, 0.5, 1.0]
    mu_effs = [0.01, 0.05, 0.1, 0.5, 1.0]
    cap_nv = 3  # large particle can only have up to 3x as many vertices as the small particle

    # add core for:
    # 20, above 0.1
    # 10, above 0.5
    # 6, above 0.5
    # 3, above 0.5
    # can figure this ^ out by getting the proper energy scale

    disks = []
    rbs = []
    parameter_combinations = []
    for nv in nvs:
        for mu_eff in mu_effs:
            print(nv, mu_eff)
            parameter_combinations.append((nv, mu_eff))
            # make a 'disk' system for each state point (its not really a disk system, just a poly particle with 0 friction)
            disks.append(build_rigid_bumpy_system_from_radii(radii, which, 0, 2, packing_fraction, False, int(time.time())))
            # make a rigid bumpy system for each state point
            add_core = (nv >= 20 and mu_eff >= 0.1) or (nv >= 3 and mu_eff > 0.5)
            rbs.append(build_rigid_bumpy_system_from_radii(radii, which, mu_eff, nv, packing_fraction, add_core, int(time.time()), cap_nv=cap_nv))
            if nv == 6 and mu_eff == 0.5:
                print(rbs[-1].pos)
    disks = join_systems(disks)

    # minimize the energy of the disk system
    script_path = os.path.join("/home/mmccraw/dev/dpmd/build/", "rigid_bumpy_equilibrate_pbc")
    # disks.set_neighbor_method(NeighborMethod.Cell)
    # set_standard_cell_list_parameters(disks, 0.3)
    disks.set_neighbor_method(NeighborMethod.Naive)  # for some reason, the cell list doesnt work with the psuedo-disk systems
    tmp_path = tempfile.mkdtemp()
    disks.save(tmp_path)
    subprocess.run([
        script_path,
        tmp_path,
        tmp_path,
    ], check=True)
    disks = load(tmp_path, location=["final", "init"])
    shutil.rmtree(tmp_path)

    # copy the positions and box sizes of the minimized disk system to the rigid bumpy systems
    rbs_final = []
    c = 0
    for disk, rb in zip(split_systems(disks), rbs):
        nv, mu_eff = parameter_combinations[c]
        c += 1
        rb.box_size = disk.box_size.copy()
        if nv == 6 and mu_eff == 0.5:
            print(rb.pos)
            print(disk.pos)
        rb.pos = disk.pos.copy()

        rb.set_vertices_on_particles_as_disk()
        rb.calculate_mu_eff()
        if nv == 6 and mu_eff == 0.5:
            print(rb.pos)
        rbs_final.append(rb)
    rbs = join_systems(rbs_final)

    rbs.save("test")

    for sid in range(rbs.n_systems()):
        nv, mu_eff = parameter_combinations[sid]
        draw_particles_frame(None, plt.gca(), rbs, system_id=sid, use_pbc=True, which='vertex', cmap_name='viridis', location=None)
        plt.savefig(f'figures/test_{nv}_{mu_eff}.png')
        plt.close()