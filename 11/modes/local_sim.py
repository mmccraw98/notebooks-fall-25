from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from numba import njit
from tqdm import tqdm
import numpy as np

@njit(cache=True, fastmath=True)
def compute_forces_numba(pos, rad, k, neighbors_i, neighbors_j, box_size):
    force = np.zeros_like(pos, dtype=pos.dtype)
    pe = np.zeros(pos.shape[0], dtype=pos.dtype)
    for i, j in zip(neighbors_i, neighbors_j):
        dy = pos[i,1] - pos[j,1]
        ny = np.round(dy / box_size[1])
        dy -= box_size[1] * ny
        dx = pos[i,0] - pos[j,0]
        nx = np.round(dx / box_size[0])
        dx -= box_size[0] * nx
        rij = np.sqrt(dx ** 2 + dy ** 2)
        sij = rad[i] + rad[j]
        overlap = sij - rij

        if overlap > 0.0:
            mag = k * overlap
            fx = mag * dx / rij
            fy = mag * dy / rij

            force[i, 0] += fx
            force[i, 1] += fy
            force[j, 0] -= fx
            force[j, 1] -= fy

            e_pair = 0.5 * k * overlap * overlap
            pe[i] += 0.5 * e_pair
            pe[j] += 0.5 * e_pair

    return force, pe


@njit(cache=True, fastmath=True)
def compute_pairwise_force_matrix(pos, rad, k, neighbors_i, neighbors_j, box_size):
    F = np.zeros((pos.shape[0], pos.shape[0], 2))
    R = np.zeros((pos.shape[0], pos.shape[0], 2))
    for i, j in zip(neighbors_i, neighbors_j):
        dy = pos[i,1] - pos[j,1]
        ny = np.round(dy / box_size[1])
        dy -= box_size[1] * ny
        dx = pos[i,0] - pos[j,0]
        nx = np.round(dx / box_size[0])
        dx -= box_size[0] * nx
        
        rij = np.sqrt(dx ** 2 + dy ** 2)
        sij = rad[i] + rad[j]
        overlap = sij - rij

        if overlap > 0.0:
            mag = k * overlap
            fx = mag * dx / rij
            fy = mag * dy / rij

            F[i, j, 0] += fx
            F[i, j, 1] += fy
            F[j, i, 0] -= fx
            F[j, i, 1] -= fy

        R[i, j, 0] = dx
        R[i, j, 1] = dy
        R[j, i, 0] = - dx
        R[j, i, 1] = - dy
    return F, R

def build_neighbor_list(pos, box_size, rmax):
    tree = cKDTree(np.mod(pos, box_size), boxsize=box_size)
    pairs = np.fromiter(tree.query_pairs(r=rmax), dtype=np.dtype([('i', np.int32), ('j', np.int32)]))
    return pairs['i'], pairs['j']