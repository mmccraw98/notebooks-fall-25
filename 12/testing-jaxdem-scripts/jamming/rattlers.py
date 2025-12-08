import numpy as np
import jax
import jax.numpy as jnp

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

def get_disk_rattlers_2d(ij, h_ij, r_ij, check_forces=True, zc=3):
    """
    Iteratively remove rattlers from a contact network, return the ids of the rattler and non-rattler particles
    Rattlers are treated as particles with fewer than 3 contacts
    Optionally, can also define rattlers through their force networks, an exact approach
    """
    ids = np.unique(ij)
    contact_mask = h_ij > 0
    ij = ij[contact_mask]
    r_ij = r_ij[contact_mask]

    rattler_ids = np.setdiff1d(ids, np.unique(ij))

    while True:
        ids = np.unique(ij)
        if len(ids) == 0:
            print('No valid pairs remaining')
            break
        num_contacts = np.bincount(ij[:, 0])
        rattler_condition = num_contacts[ids] < zc
        if check_forces:
            rattler_condition &= np.array([is_force_rattler(r_ij[ij[:, 0] == i, :]) for i in ids])
        new_rattlers = np.setdiff1d(ids[rattler_condition], rattler_ids)
        rattler_ids = np.union1d(rattler_ids, new_rattlers)
        rattler_mask = np.any(np.isin(ij, rattler_ids), axis=1)
        if len(new_rattlers) == 0:
            break
        ij = ij[~rattler_mask]
        if check_forces:
            r_ij = r_ij[~rattler_mask]
    return rattler_ids, np.unique(ij)

def get_non_rattler_state_disk_2d(state, system):
    r_ij = system.domain.displacement(state.pos[None, :, :], state.pos[:, None, :], system)
    d_ij = jnp.linalg.norm(r_ij, axis=-1)
    sigma_ij = state.rad[None, :] + state.rad[:, None]
    contact_mask = (d_ij < sigma_ij) & (d_ij > 0)
    h_ij = (sigma_ij - d_ij) * contact_mask

    self_mask = np.all(r_ij == 0, axis=-1)

    r_ij = r_ij[~self_mask].reshape(-1, 2)
    h_ij = h_ij[~self_mask].reshape(-1)

    indices = np.arange(state.N)
    i, j = np.meshgrid(indices, indices)
    ij = np.concatenate((i[..., None], j[..., None]), axis=-1)[~self_mask].reshape(-1, 2)
    ids = np.unique(ij)

    rattler_ids, non_rattler_ids = get_disk_rattlers_2d(ij, h_ij, r_ij)

    return jax.tree.map(lambda x: x[non_rattler_ids] if x.size > 1 else x, state), system