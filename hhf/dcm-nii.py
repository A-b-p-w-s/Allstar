import vtk
import vtk.util.numpy_support as ns
import numpy as np
import nibabel as nib
import sys

def convert_dicom_to_nifti(dicom_dir, output_file):
    # Read DICOM files
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
    reader.Update()
    
    # Get the VTK image data
    image_data = reader.GetOutput()
    
    # Convert VTK image data to numpy array
    point_data = image_data.GetPointData()
    array_data = point_data.GetScalars()
    numpy_data = ns.vtk_to_numpy(array_data)
    
    # Get image dimensions and reshape the numpy array
    dims = image_data.GetDimensions()
    numpy_data = numpy_data.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(2, 1, 0)
    
    # Create NIfTI image
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    
    # Save NIfTI image
    nib.save(nifti_img, output_file)

if __name__ == "__main__":

    dicom_dir = r'D:\hhf\test\dcm55'
    output_file = r'D:\hhf\test\a.nii'

    convert_dicom_to_nifti(dicom_dir, output_file)
    print(f"Conversion completed: {output_file}")
