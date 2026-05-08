import vtk
from vtk.util.numpy_support import vtk_to_numpy
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from pathlib import Path
import numpy as np

# Define your surface file
base_dir = Path.home() / "eigenmode_fingerprints" / "pang_surfaces" / "fsaverage5"
hemi_files = {
    "L": "lh.white.vtk",
    "R": "rh.white.vtk"
}

for hemi, fname in hemi_files.items():
    vtk_path = base_dir / fname
    output_gii = vtk_path.with_suffix(".surf.gii")

    if not vtk_path.exists():
        print(f"❌ File not found: {vtk_path}")
        continue

    # Read VTK file using vtkPolyDataReader
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    polydata = reader.GetOutput()

    # Extract coordinates and faces
    points = vtk_to_numpy(polydata.GetPoints().GetData())  # shape (V, 3)
    faces_vtk = polydata.GetPolys().GetData()
    faces_np = vtk_to_numpy(faces_vtk).reshape(-1, 4)[:, 1:]  # skip leading count

    # Convert and save as GIFTI
    coords_array = GiftiDataArray(data=points, intent="NIFTI_INTENT_POINTSET")
    faces_array = GiftiDataArray(data=faces_np.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")
    gii = GiftiImage(darrays=[coords_array, faces_array])
    nib.save(gii, output_gii)

    print(f"✅ Converted {fname} → {output_gii.name}")