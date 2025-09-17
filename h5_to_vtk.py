import numpy as np
import h5py
from pyevtk.hl import imageToVTK
def h5_to_vtk(h5_filename, output_basename="velocity_field", index=25):
    """
    Convert an HDF5 file containing 3D velocity components u, v, w and
    a mask to a VTK image data (.vti) file readable by ParaView.
    Parameters:
    -----------
    h5_filename : str
        Path to the input .h5 file.
    output_basename : str
        Base name (without extension) for the output VTK file.
    """
    # 1. Read the HDF5 file
    with h5py.File(h5_filename, 'r') as f:
        # Assuming the datasets are named 'u', 'v', 'w', 'mask'
        u = f["u"][index]     # shape (nx, ny, nz) = (320, 200, 250) for example
        v = f["v"][index]
        w = f["w"][index]
        u_unc = f["uncertainty_u"][index]
        v_unc = f["uncertainty_v"][index]
        w_unc = f["uncertainty_w"][index]
        mask = f[mask_name][:]
    if len(mask.shape) == 4:
        mask = mask[0]
    binary_mask = (mask != 0).astype(np.uint8)
    # 2. Decide on the image origin and spacing
    #    These are the spatial coordinates assigned to your data.
    #    For simple integer grids, you can just use (0,0,0) with spacing (1,1,1).
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0) #!# (1.0, 1.0, 1.0) HR/SR, (2.0, 2.0, 2.0) LR
    # 3. pyevtk expects data in the shape (nx, ny, nz).
    #    Make sure your arrays have that shape.
    #    If your data is truly in shape (320, 200, 250),
    #    we assume x=320, y=200, z=250 is correct.
    #    *Optional:* If you have to reorder, you could do:
    #    u = np.transpose(u, (2,1,0))  # only if your dimension ordering is different.
    #    but presumably, you do not need this if your data is already Nx x Ny x Nz.
    # 4. Write out using imageToVTK as a 3D image.
    #    We can store the velocity as a vector and mask as a scalar field.
    #    pyevtk supports vector data by passing a tuple (u, v, w).
    #    We’ll pass that as point data.
    #imageToVTK(
    #    output_basename,
    #    origin=origin,
    #    spacing=spacing,
    #    pointData={
    #        "velocity": (u, v, w),  # This will create a 3-component vector in ParaView
    #        mask_name: binary_mask            # This is just a scalar field (0 or 1)
    #    }
    #)
    imageToVTK(
    output_basename,
    origin=origin,
    spacing=spacing,
    pointData={},
    cellData={
        "velocity": (u, v, w),
        mask_name: binary_mask,
        "uncertainty_u": u_unc,
        "uncertainty_v": v_unc,
        "uncertainty_w": w_unc
    }
    )
def save_to_h5(output_filepath, col_name, dataset):
    #dataset = np.expand_dims(dataset, axis=0)
    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    with h5py.File(output_filepath, 'a') as hf:
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset

def add_mask(from_h5, to_h5, mask_name="mask_cleaned"):
    with h5py.File(from_h5, 'r') as f:
        mask = f[mask_name][:]
    # Check if mask already exists in the target file
    with h5py.File(to_h5, 'a') as f_out:
        if mask_name in f_out:
            raise ValueError(f"Dataset '{mask_name}' already exists in {to_h5}.")
    save_to_h5(to_h5, mask_name, mask)

if __name__ == "__main__":
    h5_filename = "4DFlowNet_DE/result/aorta_result_DE.h5"
    output_basename = "4DFlowNet_DE/aorta_result_DE_vtk"

    mask_name = "mask"
    #add_mask(from_h5="4DFlowNet_MCD/data/aorta_CFD/aorta03_HR.h5", to_h5=h5_filename, mask_name=mask_name)
    h5_to_vtk(h5_filename, output_basename, index=1)