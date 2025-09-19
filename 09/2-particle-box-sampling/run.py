"""
Simple investigation of the diffusion of a 2-particle system in a walled boundary.
The 2-particle system is composed of a disk and a bumpy particle (possibly a disk if desired) with a variable number of vertices and friction coefficient.
First, the maximum packing fraction is estimated by jamming n_jam_duplicates of the system from random initial positions within the box.
Once the maximum packing fraction (phi_j) is estimated, max_n_dynamics_duplicates of the system are create with random initial positions corresponding -
to packing fractions from phi_j to phi_j - max_phi_offset.  Each system is then equilibrated to a 0-overlap state and the velocities are set to the -
desired temperature.  Finally, NVE dynamics are run for n_steps steps.
The idea is to determine the translational and rotational diffusion coefficients for the bumpy (rotational) particles and their relationship to nearby -
free volume.
"""


import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import NeighborMethod, DT_INT
import matplotlib.pyplot as plt
import subprocess
from copy import deepcopy
import os
import shutil
from tqdm import tqdm
import h5py

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
def dedup_positions(data, out=None, catalog=None, atol=1e-12, rtol=1e-10, *, in_memory=True, return_blocks=False):
    pos = data.trajectory.pos
    angles = data.trajectory.angle
    delta_phi = data.delta_phi
    sys_mask = data.trajectory.pe_total == 0
    delta_phi_unique, inv_idx, delta_phi_counts = np.unique(delta_phi, return_inverse=True, return_counts=True)

    N_samples, N_systems = sys_mask.shape
    N_particles = data.system_id.shape[0]
    assert pos.shape == (N_samples, N_particles, 2)
    assert angles.shape == (N_samples, N_particles)

    # Particles per system
    order = np.argsort(data.system_id, kind='stable')
    sid_sorted = data.system_id[order]
    b = np.flatnonzero(np.r_[True, sid_sorted[1:] != sid_sorted[:-1], True])
    sys_vals = sid_sorted[b[:-1]]
    part_idx_by_system = [np.empty(0, dtype=order.dtype) for _ in range(N_systems)]
    for i, s in enumerate(sys_vals):
        part_idx_by_system[int(s)] = order[b[i]:b[i+1]]

    # Valid samples per system
    r, c = np.nonzero(sys_mask)
    samples_by_system = [np.empty(0, dtype=int) for _ in range(N_systems)]
    active_systems = []
    if r.size:
        ordc = np.argsort(c, kind='stable')
        c_sorted = c[ordc]; r_sorted = r[ordc]
        bb = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1], True])
        sys_cols = c_sorted[bb[:-1]]
        for i, s in enumerate(sys_cols):
            samples_by_system[int(s)] = r_sorted[bb[i]:bb[i+1]]
        active_systems = sys_cols.tolist()

    # Buckets per unique delta-phi present in this data
    K = len(delta_phi_unique)
    buckets_pos = [[] for _ in range(K)]
    buckets_ang = [[] for _ in range(K)]

    for s in active_systems:
        k = inv_idx[s]                           # index into delta_phi_unique
        samp = samples_by_system[s]              # (S_sys,)
        parts = part_idx_by_system[s]            # (M_sys,)
        block_pos = pos[samp[:, None], parts[None, :], :]  # (S_sys, M_sys, 2)
        block_ang = angles[samp[:, None], parts[None, :]]  # (S_sys, M_sys)
        buckets_pos[k].append(block_pos)
        buckets_ang[k].append(block_ang)

    # Aggregate into provided dict (keyed by integer catalog index)
    if out is None:
        out = {}

    if catalog is not None:
        catalog = np.asarray(catalog, dtype=float)
        # map using rounding, with nearest fallback within tolerance
        round_map = {round(float(v), 12): i for i, v in enumerate(catalog)}
        def map_key(v):
            rv = round(float(v), 12)
            if rv in round_map:
                return round_map[rv]
            idx = int(np.argmin(np.abs(catalog - float(v))))
            return idx if np.isclose(catalog[idx], float(v), rtol=rtol, atol=atol) else None
    else:
        def map_key(v):
            return None  # no catalog provided; will fallback to local k

    # Collect per-iteration blocks for streaming writes (always dict to satisfy linters)
    blocks = {}

    for k in range(K):
        if not buckets_pos[k]:
            continue
        key = map_key(delta_phi_unique[k])
        if key is None:
            key = int(k)  # fallback to local index if no catalog or no match
        pos_cat = np.concatenate(buckets_pos[k], axis=0)
        ang_cat = np.concatenate(buckets_ang[k], axis=0)
        s0 = int(np.flatnonzero(inv_idx == k)[0])
        pf_val = float(data.final.packing_fraction[s0])

        if key in out:
            if in_memory:
                if out[key]['pos'].shape[1:] != pos_cat.shape[1:]:
                    raise ValueError(f"Shape mismatch for key {key}: existing {out[key]['pos'].shape[1:]} vs new {pos_cat.shape[1:]}")
                out[key]['pos'] = np.concatenate([out[key]['pos'], pos_cat], axis=0)
                out[key]['angle'] = np.concatenate([out[key]['angle'], ang_cat], axis=0)
            out[key]['n_samples'] += N_samples * delta_phi_counts[k]
            out[key]['n_valid_samples'] += pos_cat.shape[0]
        else:
            entry = {
                'delta_phi': float(delta_phi_unique[k]),
                'packing_fraction': pf_val,
                'n_samples': N_samples * delta_phi_counts[k],
                'n_valid_samples': pos_cat.shape[0],
                'box_size': data.box_size[s0],
            }
            if in_memory:
                entry['pos'] = pos_cat
                entry['angle'] = ang_cat
            out[key] = entry

        if return_blocks:
            blocks[key] = {
                'pos': pos_cat,
                'angle': ang_cat,
                'delta_phi': float(delta_phi_unique[k]),
                'packing_fraction': pf_val,
                'box_size': data.box_size[s0],
                'n_samples': N_samples * delta_phi_counts[k],
                'n_valid_samples': pos_cat.shape[0],
            }

    return (out, blocks) if return_blocks else out

def save_state_h5(out, path):
    with h5py.File(path, 'w') as f:
        for key, v in out.items():
            g = f.create_group(str(key))
            g.create_dataset(
                'pos', data=v['pos'],
                compression='gzip', compression_opts=4, shuffle=True
            )
            g.create_dataset(
                'angle', data=v['angle'],
                compression='gzip', compression_opts=4, shuffle=True
            )
            g.attrs['delta_phi'] = float(v['delta_phi'])
            g.attrs['packing_fraction'] = float(v['packing_fraction'])
            g.attrs['n_samples'] = int(v['n_samples'])
            g.attrs['n_valid_samples'] = int(v['n_valid_samples'])
            g.create_dataset(
                'box_size', data=v['box_size'],
                compression='gzip', compression_opts=4, shuffle=True
            )

def append_state_h5(blocks, path):
    with h5py.File(path, 'a') as f:
        for key, v in blocks.items():
            name = str(int(key))
            pos_block = v['pos']
            ang_block = v['angle']
            s_add = int(v['n_samples'])
            sv_add = int(v['n_valid_samples'])
            if name in f:
                g = f[name]
                # validate shapes and append
                if g['pos'].shape[1:] != pos_block.shape[1:] or g['angle'].shape[1:] != ang_block.shape[1:]:
                    raise ValueError(f"Shape mismatch for key {key}")
                n0 = g['pos'].shape[0]
                n1 = n0 + pos_block.shape[0]
                g['pos'].resize((n1,) + g['pos'].shape[1:])
                g['pos'][n0:n1] = pos_block
                g['angle'].resize((n1,) + g['angle'].shape[1:])
                g['angle'][n0:n1] = ang_block
                # update attributes
                g.attrs['n_samples'] = int(g.attrs.get('n_samples', 0)) + s_add
                g.attrs['n_valid_samples'] = int(g.attrs.get('n_valid_samples', 0)) + sv_add
            else:
                g = f.create_group(name)
                M, coord = pos_block.shape[1], pos_block.shape[2]
                g.create_dataset(
                    'pos', data=pos_block, maxshape=(None, M, coord),
                    compression='gzip', compression_opts=4, shuffle=True
                )
                g.create_dataset(
                    'angle', data=ang_block, maxshape=(None, M),
                    compression='gzip', compression_opts=4, shuffle=True
                )
                g.attrs['delta_phi'] = float(v['delta_phi'])
                g.attrs['packing_fraction'] = float(v['packing_fraction'])
                g.attrs['n_samples'] = s_add
                g.attrs['n_valid_samples'] = sv_add
                if 'box_size' in v:
                    g.create_dataset(
                        'box_size', data=v['box_size'],
                        compression='gzip', compression_opts=4, shuffle=True
                    )

def load_state_h5(path):
    out = {}
    with h5py.File(path, 'r') as f:
        for name in f.keys():
            g = f[name]
            out[float(name)] = {
                'pos': g['pos'][:],
                'angle': g['angle'][:],
                'delta_phi': float(g.attrs['delta_phi']),
                'packing_fraction': float(g.attrs['packing_fraction']),
                'n_samples': int(g.attrs['n_samples']),
                'n_valid_samples': int(g.attrs['n_valid_samples']),
                'box_size': g['box_size'][:],
            }
    return out

if __name__ == "__main__":
    mu_effs = [0.01, 0.05, 0.1, 0.5]

    for mu_eff in mu_effs:

        data_root = f"/home/mmccraw/dev/data/09-09-25/bumpy-final/{mu_eff}"
        old_data_root = f"/home/mmccraw/dev/data/09-09-25/bumpy/{mu_eff}"
        script_root = "/home/mmccraw/dev/dpmd/build/"

        if not os.path.exists(data_root):
            os.makedirs(data_root)

        jam_data_path = os.path.join(old_data_root, "jam")
        offset_data_path = os.path.join(data_root, "offset")
        dynamics_data_path = os.path.join(data_root, "dynamics")
        box_sample_data_path = os.path.join(data_root, "box-sample")

        n_jam_duplicates = 1000

        n_phi_steps = 50
        min_phi_offset = 1e-3
        max_phi_offset = 4e-1

        max_n_dynamics_duplicates = 1000

        temperatures = [1e-5, 1e-6, 1e-7]
        n_steps = 1e7

        n_box_samples = 1e4
        n_samples_target = 1e3
        max_n_iterations = 100
        max_n_box_sample_duplicates = 10000

        n_vertices = 3
        initial_packing_fraction = 0.2
        rng_seed = 0

        rb = create_2_particle_bumpy_disk_system(n_vertices, mu_eff, initial_packing_fraction)
        rb.set_neighbor_method(NeighborMethod.Naive)

        # place n_jam_duplicates of the system randomly within the box, and jam them
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
        if phi_j - max_phi_offset < 0.1:
            max_phi_offset = phi_j - 0.1
        delta_phi = np.logspace(np.log10(min_phi_offset), np.log10(max_phi_offset), n_phi_steps - 1)
        delta_phi = np.concatenate((delta_phi, [phi_j - 0.1]))
        phi = phi_j - delta_phi

        # Stable catalog of targets and integer TODO indices
        delta_phi_all = delta_phi.copy()
        todo_idx = np.arange(delta_phi_all.size, dtype=int)

        # Output HDF5 path (streaming writes per iteration)
        h5_out_path = os.path.join(data_root, "box_sample_result.h5")
        if os.path.exists(h5_out_path):
            os.remove(h5_out_path)

        # create a block of n_phi_steps systems, each with a different phi value
        offset_data = join_systems([rb for _ in range(n_phi_steps)])
        offset_data.scale_to_packing_fraction(phi)
        offset_data.add_array(delta_phi, 'delta_phi')

        # create max_n_dynamics_duplicates / n_phi_steps duplicates of the concatenated dynamics_data
        offset_data = join_systems([offset_data for _ in range(max_n_dynamics_duplicates // n_phi_steps)])
        offset_data.set_positions(1, rng_seed)  # set random positions
        offset_data.save(offset_data_path, locations=["init"])
        print(jam_data_path)

        subprocess.run([  # equilibrate the system and save the data to the dynamics_data_path
            os.path.join(script_root, "rigid_bumpy_equilibrate_wall"),
            offset_data_path,
            dynamics_data_path,
        ], check=True)
        for temperature in temperatures:
            dynamics_data_path_temp = os.path.join(data_root, f"dynamics_{temperature}")
            if os.path.exists(dynamics_data_path_temp):
                shutil.rmtree(dynamics_data_path_temp)
            dynamics_data = load(dynamics_data_path, location=["final"])  # set the velocities and overwrite the data
            dynamics_data.set_velocities(temperature, rng_seed)
            dynamics_data.add_array(offset_data.delta_phi.copy(), 'delta_phi')
            dynamics_data.save(dynamics_data_path_temp, locations=["init"])
            subprocess.run([  # run the dynamics
                os.path.join(script_root, "nve_rigid_bumpy_wall_final"),
                dynamics_data_path_temp,
                dynamics_data_path_temp,
                str(n_steps),
                str(100),
            ], check=True)

            exit()

        # now run the initial dynamics data in the box sampling protocol to determine the free volume
        for i in tqdm(range(max_n_iterations)):  # continuously sample until target per catalog index
            if todo_idx.size == 0:
                break
            # Build systems only for remaining targets
            curr_phi = phi_j - delta_phi_all[todo_idx]
            box_sample_data = join_systems([rb for _ in range(len(todo_idx))])
            box_sample_data.scale_to_packing_fraction(curr_phi)
            box_sample_data.add_array(delta_phi_all[todo_idx], 'delta_phi')
            print(todo_idx)
            box_sample_data = join_systems([box_sample_data for _ in range(max_n_box_sample_duplicates // len(todo_idx))])
            if os.path.exists(box_sample_data_path):
                shutil.rmtree(box_sample_data_path)
            box_sample_data.save(box_sample_data_path, locations=["init"])
            subprocess.run([
                os.path.join(script_root, "box_sample_rigid_bumpy_wall_final"),
                box_sample_data_path,
                box_sample_data_path,
                str(n_box_samples),
                str(0.5 * dynamics_data.vertex_rad.max()),
                str(np.random.randint(0, 1000000)),
            ], check=True)
            box_sample_data = load(box_sample_data_path, location=["init", "final"], load_trajectory=True, load_full=True)
            state, blocks = dedup_positions(
                box_sample_data,
                None,
                catalog=delta_phi_all,
                in_memory=False,
                return_blocks=True,
            )
            append_state_h5(blocks, h5_out_path)
            # Recompute TODO by integer keys using accumulated valid counts (keep zeros)
            remaining = []
            for idx in todo_idx:
                cnt = state[idx]['n_valid_samples'] if (state is not None and idx in state) else 0
                if cnt < n_samples_target:
                    remaining.append(idx)
            todo_idx = np.array(remaining, dtype=int)
            del box_sample_data, state, blocks
        shutil.rmtree(box_sample_data_path)