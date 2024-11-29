'''
/*******************************************************/
 * Copyright © 2024 泰玛 Tarek Ali, Allstar Medical. All rights reserved.
 /*******************************************************/
 
 Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and non-commercial purposes is hereby granted, without fee and without a signed licensing agreement, provided that the above copyright notice, this paragraph, and the following disclaimer appear in all copies, modifications, and distributions.

IN NO EVENT SHALL 泰玛 Tarek Ali OR ALLSTAR MEDICAL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF 泰玛 Tarek Ali OR ALLSTAR MEDICAL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

泰玛 Tarek Ali SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND 泰玛 Tarek Ali HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
'''

import numpy as np
import pandas as pd
import os
import glob
import nibabel as nib
from pathlib import Path
from PIL import Image
import re
from fastai.basics import *
from fastai.vision import *

print('Processing images...')

###################################################
# Function to sort filenames naturally
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Walk through the directory to collect filenames
file_list = []
for root, _, filenames in os.walk('Task03_Liver/imagesTr'):
    for filename in filenames:
        file_list.append((root, filename))

# Create a DataFrame with directory and filenames
df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])

# Apply natural sorting to the filenames
df_files['filename'] = df_files['filename'].apply(str)
df_files = df_files.sort_values(by='filename', key=lambda x: x.map(natural_keys)).reset_index(drop=True)

########################################################
# Assign mask filenames based on the collected filenames
df_files['mask_dirname'] = ""
df_files["mask_filename"] = ""

for i in range(131):  # Assuming you have 131 files
    ct = f"liver_{i}.nii.gz"
    mask = f"liver_{i}.nii.gz"
    df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
    df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = "Task03_Liver/imagesTr"

# Separate files for which masks are not found
df_files_test = df_files[df_files.mask_filename == '']
df_files = df_files[df_files.mask_filename != ''].sort_values(by=['filename'], key=lambda x: x.map(natural_keys)).reset_index(drop=True)


######################
# Function to read NIfTI files
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

#######################################
# Utils
dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60)
)
########################################

    
    
@patch
def windowed(self: Tensor, w, l):
    '''
    Apply windowing to the Tensor image.

    Args:
    - w: Width of the window.
    - l: Level of the window.

    Returns:
    - Tensor: Windowed image.
    '''
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)

class TensorCTScan(TensorImageBW): _show_args = {'cmap': 'bone'}

@patch
def freqhist_bins(self: Tensor, n_bins=100):
    """
    Calculate frequency histogram bins for the Tensor image.

    Args:
    - n_bins: Number of bins.

    Returns:
    - Tensor: Histogram bins.
    """
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()

@patch
def hist_scaled(self: Tensor, brks=None):
    """
    Scale the Tensor image histogram.

    Args:
    - brks: Breakpoints for scaling.

    Returns:
    - Tensor: Scaled histogram.
    """
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    """
    Convert Tensor image to multiple channels based on windowing and histogram scaling.

    Args:
    - wins: Windows for windowing.
    - bins: Histogram bins.

    Returns:
    - TensorCTScan: Converted Tensor image.
    """
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0: res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def save_tiff_512x512(self: Tensor, path, wins, bins=None, quality=90, view='axial'):
    """
    Save Tensor image as TIFF format with specified parameters.

    Args:
    - path: Path to save the image.
    - wins: Windows for windowing.
    - bins: Histogram bins.
    - quality: Image quality.
    - view: Image view ('axial', 'coronal', 'sagittal').

    Returns:
    - None
    """
    patient_id = Path(path).parent.stem  # Get parent directory name (patient ID)
    slice_type = Path(path).stem.split('_')[-1]  # Get slice type from filename
    
    # Determine slice orientation (axial, coronal, sagittal)
    if view == 'axial':
        save_dir = f'train_images/{patient_id}/axial'
        mask_dir = f'train_masks/{patient_id}/axial'
        slices = self.shape[2]
        for i in range(slices):
            data = tensor(self[..., i].astype(np.float32))
            data.save_tiff_512x512(f"{save_dir}/{patient_id}_slice_{i}.jpg", [dicom_windows.liver, dicom_windows.custom])
            
            mask = tensor(self[..., i].astype(np.float32))
            mask.save_tiff_512x512(f"{mask_dir}/{patient_id}_mask_slice_{i}.tiff", [dicom_windows.liver, dicom_windows.custom])
    
    elif view == 'coronal':
        save_dir = f'train_images/{patient_id}/coronal'
        mask_dir = f'train_masks/{patient_id}/coronal'
        slices = self.shape[1]
        for i in range(slices):
            data = tensor(self[:, i, :].astype(np.float32))
            data.save_tiff_512x512(f"{save_dir}/{patient_id}_slice_{i}.jpg", [dicom_windows.liver, dicom_windows.custom])
            
            mask = tensor(self[:, i, :].astype(np.float32))
            mask.save_tiff_512x512(f"{mask_dir}/{patient_id}_mask_slice_{i}.tiff", [dicom_windows.liver, dicom_windows.custom])
    
    elif view == 'sagittal':
        save_dir = f'train_images/{patient_id}/sagittal'
        mask_dir = f'train_masks/{patient_id}/sagittal'
        slices = self.shape[0]
        for i in range(slices):
            data = tensor(self[i, :, :].astype(np.float32))
            data.save_tiff_512x512(f"{save_dir}/{patient_id}_slice_{i}.jpg", [dicom_windows.liver, dicom_windows.custom])
            
            mask = tensor(self[i, :, :].astype(np.float32))
            mask.save_tiff_512x512(f"{mask_dir}/{patient_id}_mask_slice_{i}.tiff", [dicom_windows.liver, dicom_windows.custom])
    
    else:
        raise ValueError(f"Invalid view type: {view}. Must be one of 'axial', 'coronal', 'sagittal'.")
    

# Generate images
slice_sum = 0
for ii in range(len(df_files)):
    ct_file = df_files.loc[ii, 'dirname'] + "/" + df_files.loc[ii, 'filename']
    mask_file = df_files.loc[ii, 'mask_dirname'] + "/" + df_files.loc[ii, 'mask_filename']
    
    ct_data = read_nii(ct_file)
    mask_data = read_nii(mask_file)
    
    # Export axial slices
    ct_tensor = tensor(ct_data)
    mask_tensor = tensor(mask_data)
    ct_tensor.save_tiff_512x512(ct_file, [dicom_windows.liver, dicom_windows.custom], view='axial')
    mask_tensor.save_tiff_512x512(mask_file, [dicom_windows.liver, dicom_windows.custom], view='axial')
    
    # Export coronal slices
    ct_tensor.save_tiff_512x512(ct_file, [dicom_windows.liver, dicom_windows.custom], view='coronal')
    mask_tensor.save_tiff_512x512(mask_file, [dicom_windows.liver, dicom_windows.custom], view='coronal')
    
    # Export sagittal slices
    ct_tensor.save_tiff_512x512(ct_file, [dicom_windows.liver, dicom_windows.custom], view='sagittal')
    mask_tensor.save_tiff_512x512(mask_file, [dicom_windows.liver, dicom_windows.custom], view='sagittal')
    
    slice_sum += ct_data.shape[2]  # Total slices

print(f"Total slices exported: {slice_sum}")
