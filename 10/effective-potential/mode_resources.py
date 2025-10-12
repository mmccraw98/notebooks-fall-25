def is_force_rattler(bond_vectors, eps=1e-9):
    """
    Determine if a particle is a rattler based off its bond vectors (force or direction vectors) with neighbors
    If the largest angular gap between adjacent bond vectors exceeds pi, it is a rattler
    """
    if len(bond_vectors) < 3:
        return True
    ang = np.mod(np.arctan2(bond_vectors[:,1], bond_vectors[:,0]), 2*np.pi)
    ang.sort()
    gaps = np.diff(np.r_[ang, ang[0]+2*np.pi])
    return gaps.max() > np.pi - eps

def get_rigid_bumpy_rattlers(pair_forces, pair_ids, pair_vertex_contacts, zc=4, check_forces=False):
    """
    Iteratively remove rattlers from a contact network, return the ids of the rattler and non-rattler particles
    Rattlers are treated as particles with fewer than 4 vertices in contact with neighboring particles
    Optionally, can also define rattlers through their force networks, an exact approach
    """
    all_particle_ids = np.unique(pair_ids)

    force_norm = np.linalg.norm(pair_forces, axis=1)
    contact_mask = force_norm > 0
    pair_ids = pair_ids[contact_mask]
    pair_vertex_contacts = pair_vertex_contacts[contact_mask]
    if check_forces:
        pair_forces = pair_forces[contact_mask]

    valid_particles = np.unique(pair_ids)
    rattler_ids = np.setdiff1d(all_particle_ids, valid_particles)
    while True:
        particle_ids = np.unique(pair_ids[:, 0])
        if len(particle_ids) == 0:
            print('No valid pairs remaining')
            break
        num_vertex_contacts = np.bincount(pair_ids[:, 0], weights=pair_vertex_contacts)
        rattler_condition = num_vertex_contacts[particle_ids] < zc
        if check_forces:
            rattler_condition &= np.array([is_force_rattler(pair_forces[pair_ids[:, 0] == i, :]) for i in particle_ids])
        new_rattlers = np.setdiff1d(particle_ids[rattler_condition], rattler_ids)
        rattler_ids = np.union1d(rattler_ids, new_rattlers)
        rattler_mask = np.any(np.isin(pair_ids, rattler_ids), axis=1)
        if len(new_rattlers) == 0:
            break
        pair_ids = pair_ids[~rattler_mask]
        pair_vertex_contacts = pair_vertex_contacts[~rattler_mask]
        if check_forces:
            pair_forces = pair_forces[~rattler_mask]
    return rattler_ids, np.unique(pair_ids[~rattler_mask])

def get_disk_rattlers(pair_forces, pair_ids, zc=3, check_forces=False):
    """
    Iteratively remove rattlers from a contact network, return the ids of the rattler and non-rattler particles
    Rattlers are treated as particles with fewer than 3 contacts
    Optionally, can also define rattlers through their force networks, an exact approach
    """
    all_particle_ids = np.unique(pair_ids)

    force_norm = np.linalg.norm(pair_forces, axis=1)
    contact_mask = force_norm > 0
    pair_ids = pair_ids[contact_mask]
    if check_forces:
        pair_forces = pair_forces[contact_mask]

    valid_particles = np.unique(pair_ids)
    rattler_ids = np.setdiff1d(all_particle_ids, valid_particles)
    while True:
        particle_ids = np.unique(pair_ids[:, 0])
        if len(particle_ids) == 0:
            print('No valid pairs remaining')
            break
        num_contacts = np.bincount(pair_ids[:, 0])
        rattler_condition = num_contacts[particle_ids] < zc
        if check_forces:
            rattler_condition &= np.array([is_force_rattler(pair_forces[pair_ids[:, 0] == i, :]) for i in particle_ids])
        new_rattlers = np.setdiff1d(particle_ids[rattler_condition], rattler_ids)
        rattler_ids = np.union1d(rattler_ids, new_rattlers)
        rattler_mask = np.any(np.isin(pair_ids, rattler_ids), axis=1)
        if len(new_rattlers) == 0:
            break
        pair_ids = pair_ids[~rattler_mask]
        if check_forces:
            pair_forces = pair_forces[~rattler_mask]
    return rattler_ids, np.unique(pair_ids[~rattler_mask])


def get_dynamical_matrix_modes_for_rigid_bumpy(data, critical_rattler_contact_count=4, use_forces_for_rattler_check=True):
    hess_dim = 3  # for rigid particles

    hessian_block = [
        [data.final.hessian_xx, data.final.hessian_xy, data.final.hessian_xt],
        [data.final.hessian_yx, data.final.hessian_yy, data.final.hessian_yt],
        [data.final.hessian_tx, data.final.hessian_ty, data.final.hessian_tt]
    ]

    H_list, M_list, vec_list, val_list, non_rattler_id_list = [], [], [], [], []

    for sid in range(data.n_systems()):
        pid_0 = data.system_offset[sid]
        pid_N = data.system_offset[sid + 1]
        N = data.system_size[sid]

        mask = np.all((data.final.pair_ids >= pid_0) & (data.final.pair_ids < pid_N), axis=1)
        pair_ids = data.final.pair_ids[mask]
        local_pair_ids = pair_ids - pid_0
        pair_forces = data.final.pair_forces[mask]

        # remove rattlers
        rattler_ids, non_rattler_ids = get_rigid_bumpy_rattlers(
            data.final.pair_forces[mask].copy(),
            local_pair_ids.copy(),
            data.final.pair_vertex_contacts[mask, 0].copy(),
            critical_rattler_contact_count,
            check_forces=use_forces_for_rattler_check
        )

        # calculate hessian for non-rattlers
        N = non_rattler_ids.size
        H = np.zeros((N, N, hess_dim, hess_dim))
        for pair_id, (i, j) in enumerate(local_pair_ids):
            if i in rattler_ids or j in rattler_ids:
                continue
            i = np.where(non_rattler_ids == i)[0][0]
            j = np.where(non_rattler_ids == j)[0][0]
            for a, hessian_row in enumerate(hessian_block):
                for b, hessian_term in enumerate(hessian_row):
                    # diagonal term
                    H[i, i, a, b] += hessian_term[mask][pair_id, 0]
                    # off-diagonal term
                    H[i, j, a, b] += hessian_term[mask][pair_id, 1]

        H = H.transpose(2, 0, 3, 1).reshape(hess_dim * N, hess_dim * N)
        M_diag = [data.mass[pid_0: pid_N], data.mass[pid_0: pid_N], data.moment_inertia[pid_0: pid_N]]
        M = np.diag(np.concatenate([m[non_rattler_ids] for m in M_diag]))

        # compute dynamical matrix modes
        vals, vecs = sp.linalg.eigh(H, M)

        assert np.allclose(H, H.T)

        H_list.append(H)
        M_list.append(M)
        val_list.append(vals)
        vec_list.append(vecs)
        non_rattler_id_list.append(non_rattler_ids)
    return H_list, M_list, val_list, vec_list, non_rattler_id_list

def get_dynamical_matrix_modes_for_disk(data, critical_rattler_contact_count=3, use_forces_for_rattler_check=True):
    hess_dim = 2  # for disks

    hessian_block = [
        [data.final.hessian_xx, data.final.hessian_xy],
        [data.final.hessian_yx, data.final.hessian_yy],
    ]

    H_list, M_list, vec_list, val_list, non_rattler_id_list = [], [], [], [], []

    for sid in tqdm(range(data.n_systems())):
        pid_0 = data.system_offset[sid]
        pid_N = data.system_offset[sid + 1]
        N = data.system_size[sid]

        mask = np.all((data.final.pair_ids >= pid_0) & (data.final.pair_ids < pid_N), axis=1)
        pair_ids = data.final.pair_ids[mask]
        local_pair_ids = pair_ids - pid_0
        pair_forces = data.final.pair_forces[mask]

        # remove rattlers
        rattler_ids, non_rattler_ids = get_disk_rattlers(
            data.final.pair_forces[mask].copy(),
            local_pair_ids.copy(),
            critical_rattler_contact_count,
            check_forces=use_forces_for_rattler_check
        )

        # calculate hessian for non-rattlers
        N = non_rattler_ids.size
        H = np.zeros((N, N, hess_dim, hess_dim))
        for pair_id, (i, j) in enumerate(local_pair_ids):
            if i in rattler_ids or j in rattler_ids:
                continue
            i = np.where(non_rattler_ids == i)[0][0]
            j = np.where(non_rattler_ids == j)[0][0]
            for a, hessian_row in enumerate(hessian_block):
                for b, hessian_term in enumerate(hessian_row):
                    # diagonal term
                    H[i, i, a, b] += hessian_term[mask][pair_id, 0]
                    # off-diagonal term
                    H[i, j, a, b] += hessian_term[mask][pair_id, 1]

        H = H.transpose(2, 0, 3, 1).reshape(hess_dim * N, hess_dim * N)
        M_diag = [data.mass[pid_0: pid_N], data.mass[pid_0: pid_N]]
        M = np.diag(np.concatenate([m[non_rattler_ids] for m in M_diag]))

        # compute dynamical matrix modes
        vals, vecs = sp.linalg.eigh(H, M)

        assert np.allclose(H, H.T)

        H_list.append(H)
        M_list.append(M)
        val_list.append(vals)
        vec_list.append(vecs)
        non_rattler_id_list.append(non_rattler_ids)
    return H_list, M_list, val_list, vec_list, non_rattler_id_list