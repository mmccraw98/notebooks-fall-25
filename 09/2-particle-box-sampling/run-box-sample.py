import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import NeighborMethod, DT_INT
import subprocess
import os
import shutil
from tqdm import tqdm
from box_sampling_resources import build_domains

# create bumpy - disk 2-particle system
def create_2_particle_bumpy_disk_system(n_vertices: int, mu_eff: float, packing_fraction: float):
    n_vertices_per_particle = np.array([n_vertices, 1], dtype=DT_INT)
    particle_radius = 0.5
    particle_mass = 1.0
    e_interaction = 1.0

    rb = RigidBumpy()
    rb.allocate_particles(n_vertices_per_particle.size)
    rb.allocate_systems(1)
    rb.allocate_vertices(n_vertices_per_particle.sum())
    rb.n_vertices_per_particle = n_vertices_per_particle
    rb.set_ids()
    rb.validate()
    rb.rad.fill(particle_radius)
    rb.mass.fill(particle_mass)
    rb.e_interaction.fill(e_interaction)
    rb.calculate_uniform_vertex_mass()

    if n_vertices > 1:
        vertex_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, particle_radius, n_vertices)
        rb.vertex_rad.fill(vertex_radius)
        rb.vertex_rad[-1] = particle_radius
    else:
        rb.vertex_rad.fill(particle_radius)

    rb.box_size.fill(1.0)
    rb.set_positions(0, 0)
    rb.set_vertices_on_particles_as_disk()
    rb.calculate_inertia()
    rb.scale_to_packing_fraction(packing_fraction)

    return rb

if __name__ == "__main__":
    mu_effs = [0.01, 0.05, 0.1, 0.5]

    for mu_eff in mu_effs:

        data_root = f"/home/mmccraw/dev/data/09-09-25/box-sample/bumpy/{mu_eff}"
        script_root = "/home/mmccraw/dev/dpmd/build/"

        if not os.path.exists(data_root):
            os.makedirs(data_root)

        jam_data_path = os.path.join(data_root, "jam")
        box_sample_data_path = os.path.join(data_root, "box-sample-large-box")
        if not os.path.exists(jam_data_path):
            os.makedirs(jam_data_path)
        if not os.path.exists(box_sample_data_path):
            os.makedirs(box_sample_data_path)

        n_jam_duplicates = 1000

        n_phi_steps = 50
        min_phi = 0.001
        min_phi_offset = 1e-1

        n_box_samples = 1e3
        n_samples_target = 1e3
        max_n_iterations = 1000
        max_n_box_sample_duplicates = 100000

        target_valid_fraction_min = 0.45
        target_valid_fraction_max = 0.55

        n_vertices = 3
        initial_packing_fraction = 0.2
        rng_seed = 0

        rb = create_2_particle_bumpy_disk_system(n_vertices, mu_eff, initial_packing_fraction)
        rb.set_neighbor_method(NeighborMethod.Naive)

        # # place n_jam_duplicates of the system randomly within the box, and jam them
        # jam_data = join_systems([rb for _ in range(n_jam_duplicates)])
        # jam_data.set_positions(1, rng_seed)  # set random positions
        # jam_data.set_vertices_on_particles_as_disk()  # update the vertex positions  # may not need this anymore
        # jam_data.save(jam_data_path, locations=["init"], save_trajectory=False)
        # subprocess.run([
        #     os.path.join(script_root, "jam_rigid_bumpy_wall_final"),
        #     jam_data_path,
        #     jam_data_path,
        # ], check=True)
        jam_data = load(jam_data_path, location=["final", "init"])

        # find the unique phi_j values and pick the highest one
        pe_tol = 1e-15
        pe_mask = jam_data.final.pe_total / jam_data.system_size < pe_tol
        phi_j = np.max(jam_data.final.packing_fraction[pe_mask])
        max_phi_offset = phi_j - min_phi
        delta_phi = np.logspace(np.log10(min_phi_offset), np.log10(max_phi_offset), n_phi_steps)
        phi = phi_j - delta_phi

        # copy the positions and angles of the largest packing fraction system
        largest_packing_fraction_system = split_systems(jam_data)[np.argmax(jam_data.packing_fraction)]
        rb.pos = largest_packing_fraction_system.pos
        rb.angle = largest_packing_fraction_system.angle
        rb.vertex_pos = largest_packing_fraction_system.vertex_pos
        rb.box_size = largest_packing_fraction_system.box_size

        # create a block of n_phi_steps systems, each with a different phi value
        bs_data = join_systems([rb for _ in range(n_phi_steps)])
        bs_data.scale_to_packing_fraction(phi)
        bs_data.add_array(delta_phi, 'delta_phi')

        # create max_n_box_sample_duplicates / n_phi_steps duplicates of the concatenated offset_data
        bs_data = join_systems([bs_data for _ in range(max_n_box_sample_duplicates // n_phi_steps)])
        domain_length = 10000.0 * np.ones(bs_data.pos.shape[0])
        build_domains(bs_data, domain_style="square", domain_kwargs={"domain_length": domain_length}, clamp_to_box=True)
        shutil.rmtree(box_sample_data_path)
        bs_data.save(box_sample_data_path, locations=["init"])
        np.save(os.path.join(box_sample_data_path, "domain_area.npy"), bs_data.domain_area)  # the data saving still isnt fully working so we need to manually save this

        subprocess.run([
            os.path.join(script_root, "domain_sample_rigid_bumpy_wall_final"),
            box_sample_data_path,
            box_sample_data_path,
            str(n_box_samples),
            str(np.random.randint(0, 1000000)),
        ], check=True)
        
        # for i in range(max_n_iterations):

        #     init_data = load(init_data_path, location=["init"], load_trajectory=True)
        #     # measure the fraction of valid samples for each delta_phi
        #     unique_delta_phi, delta_phi_index = np.unique(init_data.delta_phi, return_inverse=True)
        #     valid_fraction = np.array([np.mean(init_data.trajectory.pe_total[:, init_data.delta_phi == p] == 0) for p in unique_delta_phi])
        #     valid_fraction_by_particle = (valid_fraction[delta_phi_index])[init_data.system_id]
        #     # if the valid fraction is greater than the target_valid_fraction_max, the domain_length is too large
        #     domain_length[valid_fraction_by_particle > target_valid_fraction_max] *= 1.1
        #     # if the valid fraction is less than the target_valid_fraction_min, the domain_length is too small
        #     domain_length[valid_fraction_by_particle < target_valid_fraction_min] *= 0.9
        #     # otherwise, we are done
        #     if np.all(valid_fraction >= target_valid_fraction_min) and np.all(valid_fraction <= target_valid_fraction_max):
        #         break

        #     rb.pos = largest_packing_fraction_system.pos
        #     rb.angle = largest_packing_fraction_system.angle
        #     rb.vertex_pos = largest_packing_fraction_system.vertex_pos
        #     rb.box_size = largest_packing_fraction_system.box_size

        #     # create a block of n_phi_steps systems, each with a different phi value
        #     init_data = join_systems([rb for _ in range(n_phi_steps)])
        #     init_data.scale_to_packing_fraction(phi)
        #     init_data.add_array(delta_phi, 'delta_phi')

        #     # create max_n_box_sample_duplicates / n_phi_steps duplicates of the concatenated offset_data
        #     init_data = join_systems([init_data for _ in range(max_n_box_sample_duplicates // n_phi_steps)])
        #     build_domains(init_data, domain_style="square", domain_kwargs={"domain_length": domain_length})
        #     shutil.rmtree(init_data_path)
        #     init_data.save(init_data_path, locations=["init"])

