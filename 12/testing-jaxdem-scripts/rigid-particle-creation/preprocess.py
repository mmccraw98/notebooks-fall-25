import sys
import os
import numpy as np
import h5py
import vtk
from vtk.util import numpy_support

def save_spheres_vtp(filename, positions, radii, particle_ids=None):
    """
    Writes particle positions, radii, and IDs to a .vtp file.
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.h5 output.vtp", file=sys.stderr)
        sys.exit(1)

    h5_path = sys.argv[1]
    vtp_path = sys.argv[2]

    if not os.path.exists(h5_path):
        print(f"Error: Input file {h5_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load H5 data
    with h5py.File(h5_path, 'r') as f:
        pos = f['pos'][:]
        rad = f['rad'][:]
        if 'ID' in f:
            ids = f['ID'][:]
        else:
            ids = None
        box_size = f['box_size'][:]

    dim = pos.shape[1]
    
    # Apply PBC wrapping
    pos = np.mod(pos, box_size)
    
    # Save VTP
    save_spheres_vtp(vtp_path, pos, rad, ids)
    
    # Print metadata for the render script to standard output
    # Format: DIM BOX_X BOX_Y [BOX_Z]
    box_str = " ".join(map(str, box_size))
    print(f"{dim} {box_str}")

if __name__ == "__main__":
    main()

