import SimpleITK as sitk
import os
from tqdm import tqdm

def convert_nifti_to_dicom(nifti_file, dicom_output_dir):

    entries = os.listdir(nifti_file)
    
    # 筛选出以.nii.gz结尾的文件
    nifti_gz_files = [file for file in entries if file.endswith('.nii.gz')]
    id_i=0

    for dir in tqdm(nifti_gz_files):
        input_dir=os.path.join(nifti_file,dir)

        # Read the NIfTI file
        nifti_image = sitk.ReadImage(input_dir)
        nifti_image = sitk.Cast(nifti_image, sitk.sitkInt16)
        # Get image size, spacing, and direction
        size = nifti_image.GetSize()
        spacing = nifti_image.GetSpacing()
        direction = nifti_image.GetDirection()
        origin = nifti_image.GetOrigin()
        
        # Create the DICOM series writer
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        # Get the number of slices in the NIfTI image
        num_slices = size[2]
        
        # Make sure the output directory exists
        if not os.path.exists(dicom_output_dir):
            os.makedirs(dicom_output_dir)
        
        # Define some basic DICOM tags (you can customize these as needed)
        tags = {
            "0010|0010": "Test^Patient",        # Patient Name
            "0020|000D": "1.2.3.4",             # Study Instance UID
            "0020|000E": "1.2.3.4.5",           # Series Instance UID
            "0008|0020": "20240101",            # Study Date
            "0008|0030": "090000",              # Study Time
            "0008|0050": "123456",              # Accession Number
            "0008|0060": "MR",                  # Modality
            "0020|0011": "1",                   # Series Number
            "0008|103E": "Test Series"          # Series Description
        }
        
        for i in range(num_slices):
            # Extract the ith slice from the NIfTI image
            slice_i = nifti_image[:, :, i]
            #-------------------------------------------------------------
            # slice_i = sitk.GetArrayFromImage(slice_i)
            # slice_i = sitk.GetImageFromArray(slice_i)
            #-------------------------------------------------------------
            
            # os.makedirs(os.path.join(dicom_output_dir, str(id_i)), exist_ok=True)
            
            # Set the file name for the DICOM file
            dicom_file = os.path.join(dicom_output_dir,f"slice_{i:04d}.dcm")
            
            # Set the DICOM tags
            for tag, value in tags.items():
                slice_i.SetMetaData(tag, value)
            
            # Update Instance Number and Image Position for each slice
            slice_i.SetMetaData("0020|0013", str(i + 1))  # Instance Number
            slice_i.SetMetaData("0020|0032", "\\".join(map(str, [
                origin[0],
                origin[1],
                origin[2] + i * spacing[2]
            ])))  # Image Position (Patient)
            slice_i.SetMetaData("0020|0037", "\\".join(map(str, direction[:6])))  # Image Orientation (Patient)
            slice_i.SetMetaData("0018|0050", str(spacing[2]))  # Slice Thickness
            slice_i.SetMetaData("0018|0088", str(spacing[2]))  # Spacing Between Slices
            
            # Write the DICOM file
            writer.SetFileName(dicom_file)
            writer.Execute(slice_i)
        
        id_i+=1

    print(f"Conversion completed: {dicom_output_dir}")

if __name__ == "__main__":
    
    nifti_file = r'C:\Users\allstar\Desktop\aaaaa\vessel-213'
    dicom_output_dir = r'C:\Users\allstar\Desktop\aaaaa\new-213\vessel-213'

    convert_nifti_to_dicom(nifti_file, dicom_output_dir)
