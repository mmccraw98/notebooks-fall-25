import os
from typing import Optional, Tuple

import numpy as np
import h5py
import vtk
from vtk.util import numpy_support


def save_spheres_vtp(filename: str, positions: np.ndarray, radii: np.ndarray, particle_ids=None) -> None:
    """
    Writes particle positions, radii, and IDs to a .vtp file.

    This is intentionally kept consistent with the single-frame renderer's preprocessing.
    """
    if os.path.exists(filename):
        os.remove(filename)

    pos = np.array(positions)
    rad = np.array(radii)

    if particle_ids is None:
        p_ids = np.arange(len(pos), dtype=np.int32)
    else:
        p_ids = np.array(particle_ids, dtype=np.int32)

    # 1. Setup Points
    points = vtk.vtkPoints()
    # Ensure 3D points for VTP
    if pos.shape[1] == 2:
        pos_3d = np.column_stack((pos, np.zeros(pos.shape[0])))
    else:
        pos_3d = pos
    points.SetData(numpy_support.numpy_to_vtk(pos_3d, deep=True))

    # 2. PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # 3. Add Data Arrays
    point_data = polydata.GetPointData()

    # Diameter
    diameters = rad * 2.0
    scale_array = numpy_support.numpy_to_vtk(diameters, deep=True)
    scale_array.SetName("Diameter")
    point_data.AddArray(scale_array)

    # Radius
    rad_array = numpy_support.numpy_to_vtk(rad, deep=True)
    rad_array.SetName("Radius")
    point_data.AddArray(rad_array)

    # Particle ID
    id_array = numpy_support.numpy_to_vtk(p_ids, deep=True)
    id_array.SetName("ParticleID")
    point_data.AddArray(id_array)

    point_data.SetActiveScalars("ParticleID")

    # 4. Write
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    writer.Write()


def _get_frame_count(pos: np.ndarray) -> int:
    # pos expected shape:
    # - (T, N, dim) for animated data
    # - (N, dim) for single-frame data (treated as T=1)
    if pos.ndim == 3:
        return int(pos.shape[0])
    if pos.ndim == 2:
        return 1
    raise ValueError(f"Expected 'pos' to have 2 or 3 dims; got shape {pos.shape}")

def load_h5_metadata(h5_path: str) -> Tuple[int, np.ndarray]:
    """
    Returns:
      dim, box_size (shape (dim,))

    For animated box_size, uses box_size[0].
    """
    with h5py.File(h5_path, "r") as f:
        pos = f["pos"]
        box_size = f["box_size"][...]

    dim = int(pos.shape[-1])
    box = np.array(box_size)
    if box.ndim == 2:
        box = box[0]
    return dim, box


def write_frame_vtp(
    h5_path: str,
    frame_idx: int,
    vtp_path: str,
) -> Tuple[int, np.ndarray]:
    """
    Writes a single timestep from an H5 trajectory to a VTP file and returns (dim, box_size).

    Expected datasets:
      - pos: (T, N, dim) or (N, dim)
      - rad: (T, N) or (N,)
      - ID (optional): (T, N) or (N,)
      - box_size: (dim,) or (T, dim)
    """
    with h5py.File(h5_path, "r") as f:
        pos_ds = f["pos"]
        rad_ds = f["rad"]
        ids_ds = f["ID"] if "ID" in f else None
        box_ds = f["box_size"]

        # Normalize shapes for single-frame inputs
        if len(pos_ds.shape) == 2:
            dim = int(pos_ds.shape[1])
            pos_t = pos_ds[...]

            box = np.array(box_ds[...])
            if box.ndim == 2:
                box = box[0]

            rad_t = rad_ds[...] if len(rad_ds.shape) == 1 else rad_ds[0]
            ids_t = None
            if ids_ds is not None:
                ids_t = ids_ds[...] if len(ids_ds.shape) == 1 else ids_ds[0]
        else:
            dim = int(pos_ds.shape[2])
            pos_t = pos_ds[frame_idx, ...]

            box = np.array(box_ds[...])
            if box.ndim == 2:
                box = box_ds[frame_idx, ...]

            rad_t = rad_ds[...] if len(rad_ds.shape) == 1 else rad_ds[frame_idx, ...]
            ids_t = None
            if ids_ds is not None:
                ids_t = ids_ds[...] if len(ids_ds.shape) == 1 else ids_ds[frame_idx, ...]

    # Apply PBC wrapping
    pos_t = np.mod(pos_t, box)

    save_spheres_vtp(vtp_path, pos_t, rad_t, ids_t)
    return dim, box


def compute_sampled_frame_indices(total_frames: int, requested_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if requested_frames <= 0:
        raise ValueError(f"requested_frames must be > 0, got {requested_frames}")
    if total_frames <= requested_frames:
        return np.arange(total_frames, dtype=int)
    # Evenly sample across [0, total_frames-1] inclusive
    return np.unique(np.round(np.linspace(0, total_frames - 1, requested_frames)).astype(int))


def get_total_frames(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        pos_shape = f["pos"].shape
    if len(pos_shape) == 3:
        return int(pos_shape[0])
    if len(pos_shape) == 2:
        return 1
    raise ValueError(f"Expected 'pos' to have 2 or 3 dims; got shape {pos_shape}")


