import numpy as np
import scipy as sp
from tqdm import tqdm

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
    return rattler_ids, np.unique(pair_ids)

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
    return rattler_ids, np.unique(pair_ids)


def get_dynamical_matrix_modes_for_rigid_bumpy(data, critical_rattler_contact_count=4, use_forces_for_rattler_check=True):
    hess_dim = 3  # for rigid particles

    hessian_block = [
        [data.final.hessian_xx, data.final.hessian_xy, data.final.hessian_xt],
        [data.final.hessian_yx, data.final.hessian_yy, data.final.hessian_yt],
        [data.final.hessian_tx, data.final.hessian_ty, data.final.hessian_tt]
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
        rattler_ids, non_rattler_ids = get_rigid_bumpy_rattlers(
            data.final.pair_forces[mask].copy(),
            local_pair_ids.copy(),
            # data.final.pair_vertex_contacts[mask, 0].copy(),  # TODO: this MAY be max of both contacts!
            np.max(data.final.pair_vertex_contacts[mask].copy(), axis=1),  # TODO: this MAY be max of both contacts!
            critical_rattler_contact_count,
            check_forces=use_forces_for_rattler_check
        )

        # calculate hessian for non-rattlers
        N = non_rattler_ids.size
        if N == 0:
            H = []
            M = []
            vals = []
            vecs = []
        else:
            H = np.zeros((N, N, hess_dim, hess_dim))
            for pair_id, (i, j) in enumerate(local_pair_ids):
                if i not in non_rattler_ids or j not in non_rattler_ids:
                    continue
                try:
                    i = np.where(non_rattler_ids == i)[0][0]
                    j = np.where(non_rattler_ids == j)[0][0]
                except:
                    print(non_rattler_ids.size, j, j in non_rattler_ids)
                    raise ValueError("ASDASDASDA")
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
        if N == 0:
            H = []
            M = []
            vals = []
            vecs = []
        else:
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


def get_eimask(_vals, eps=1e-12):
    vals = np.abs(_vals.copy())
    order = np.argsort(vals)
    vals = vals[order]
    min_val = max(vals[np.argmax(vals[1:] / vals[:-1])], vals[-1] * eps)
    return vals > min_val

def get_S_modes_from_matrices(M, C, V):
    # approach:
    # use the eigenvectors of C to compute the non-null space
    # project C, V, and K into the non-null space
    # form the matrix S = V C^{-1}
    # compare S and its eigendecomposition to that of M^{-1} K in the full space
    # evaluates the relationship: V C^{-1} = M^{-1} K

    # eigendecompose C
    # Pc, Yc = np.linalg.eigh(C @ M)
    # this causes issues because the eigenvectors in eigh are orthogonalized in the euclidean sense
    # when they need to be M orthogonal
    # we have to take this stupid roundabout approach:
    Msqrt = M ** 0.5
    Pc, Zc = np.linalg.eigh(Msqrt @ C @ Msqrt)
    Yc = np.linalg.inv(Msqrt) @ Zc

    # note that it is equivalent to the eigenproblem we are interested in: CM y = p y
    # y = M^-1/2 z
    # M^1/2 C M^1/2 z = p z
    # z = M^1/2 y
    # M^1/2 C M^1/2 M^1/2 y = p M^1/2 y
    # C M y = p y
    mask = get_eimask(Pc)
    Pc = Pc[mask]
    Yc = Yc[:, mask]

    # verify algebraic properties
    # assert np.allclose(Yc.T @ M @ Yc, np.eye(Yc.shape[1]))  # Y^T M Y = I
    # proj = Yc @ Yc.T @ M  # P = Y Y^T M  - projector
    # assert np.allclose(proj @ Yc, Yc)  # P y = Y Y^T M y = Y \delta_y = y

    # project relevant matrices into the subspace formed by C
    Cc = Yc.T @ M @ C @ M @ Yc  # diag(p_i)
    Vc = Yc.T @ M @ V @ M @ Yc  # diag(p_i w_i²)
    # Kc = Yc.T @ K @ Yc
    # Mc = Yc.T @ M @ Yc

    # assert np.allclose(Mc, np.eye(sum(mask)))  # in the subspace, everything is unweighted by mass, M == I

    # eigendecompose the S matrix by inverting C in the subspace of K - it should be stable!
    Sc = Vc @ np.linalg.inv(Cc)
    Lsc, Ysc = np.linalg.eigh(Sc)

    # lift Yc back to the full space
    Ysc_full = Yc @ Ysc
    # assert np.allclose(Ysc_full.T @ M @ Ysc_full, np.eye(sum(mask)))  # check Ysc^T M Ysc = I if Ys are truly ei-vecs of K, M

    return Lsc, Ysc_full, Sc