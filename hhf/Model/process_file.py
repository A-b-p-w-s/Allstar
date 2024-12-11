import numpy as np
import pandas as pd
import os
import re
import torch
import nibabel as nib
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from fastai.basics import *
from fastai.vision import *
from PIL import Image

print('Processing image...')

# Function to read NIfTI files
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array.copy()  # Ensure no negative stride by making a copy

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

@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)

class TensorCTScan(TensorImageBW): _show_args = {'cmap': 'bone'}

@patch
def freqhist_bins(self: Tensor, n_bins=100):
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()

@patch
def hist_scaled(self: Tensor, brks=None):
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)

@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0: res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def save_tiff_512x512(self: Tensor, path, wins, bins=None, quality=90, view='axial'):
    # Ensure parent directories exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    patient_id = Path(path).parent.stem  # Get parent directory name (patient ID)
    slice_type = Path(path).stem.split('_')[-1]  # Get slice type from filename

    if len(self.shape) == 3:  # Assuming 3D tensor (height x width x channels)
        if view == 'axial':
            save_dir = f'train_images/{patient_id}/axial'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(self.shape[0]):
                data = self[i].float() * 255  # Scale to [0, 255]
                data = data.byte()  # Convert to byte tensor
                Image.fromarray(data.numpy()).save(f"{save_dir}/{patient_id}_slice_{i}.jpg", quality=quality)
        elif view == 'coronal':
            save_dir = f'train_images/{patient_id}/coronal'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(self.shape[1]):
                data = self[:, i, :].float() * 255  # Scale to [0, 255]
                data = data.byte()  # Convert to byte tensor
                Image.fromarray(data.numpy()).save(f"{save_dir}/{patient_id}_slice_{i}.jpg", quality=quality)
        elif view == 'sagittal':
            save_dir = f'train_images/{patient_id}/sagittal'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(self.shape[2]):
                data = self[:, :, i].float() * 255  # Scale to [0, 255]
                data = data.byte()  # Convert to byte tensor
                Image.fromarray(data.numpy()).save(f"{save_dir}/{patient_id}_slice_{i}.jpg", quality=quality)
        else:
            raise ValueError(f"Invalid view type: {view}. Must be one of 'axial', 'coronal', 'sagittal'.")
    elif len(self.shape) == 2:  # Assuming 2D tensor (height x width)
        save_dir = f'train_images/{patient_id}/axial'  # Default to 'axial' view for 2D tensor
        os.makedirs(save_dir, exist_ok=True)
        data = self.float() * 255  # Scale to [0, 255]
        data = data.byte()  # Convert to byte tensor
        Image.fromarray(data.numpy()).save(f"{save_dir}/{patient_id}_{slice_type}.jpg", quality=quality)
    else:
        raise ValueError("Unsupported tensor shape. Expected 2D (height x width) or 3D (height x width x channels).")

# Define the save_tiffgray_512x512 function
@patch
def save_tiffgray_512x512(x: Tensor, path, wins, bins=None, quality=90):
    # Ensure parent directories exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fn = Path(path).with_suffix('.tiff')

    # Assuming x is a tensor with shape (C, H, W) where C is number of channels
    x = (x.to_nchan(wins, bins) * 255).byte()

    # Convert to PIL image
    if x.shape[0] == 4:
        mode = 'CMYK'
    else:
        mode = 'RGB'
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=mode)

    # Rotate the image by 270 degrees
    im = im.rotate(angle=270)

    # Convert image to grayscale using 'grayscale R' formula
    im_gray_np = np.array(im)
    im_gray_np = (0.2989 * im_gray_np[:, :, 0] +
                  0.5870 * im_gray_np[:, :, 1] +
                  0.1140 * im_gray_np[:, :, 2]).astype(np.uint8)
    im_gray_np = np.clip(im_gray_np * 1.8, 0, 255).astype(np.uint8)  # Adjust brightness factor (1.8 here)

    # Set very dark pixels to black (background)
    im_gray_np[im_gray_np < 100] = 0  # Adjust threshold as needed

    # Convert numpy array back to PIL Image
    im_gray = Image.fromarray(im_gray_np)

    # Save the grayscale image
    im_gray.save(fn, quality=quality)

# Function to calculate CNR
def calculate_cnr(image, mask):
    signal = np.mean(image[mask > 0])
    noise = np.std(image[mask == 0])
    cnr = signal / noise
    return cnr

# Specify paths for the single CT and mask files using raw strings and forward slashes
ct_file = r'D:/Liver_tarek/patient_1/4 Non Contrast  1.5  B30f.nii.gz'
mask_file = r'D:/Liver_tarek/patient_1/4 Non Contrast  1.5  B30f.nii.gz'

# Read the CT scan and mask
ct_data = read_nii(ct_file)
mask_data = read_nii(mask_file)

# Ensure non-negative stride for the NumPy arrays
ct_data = ct_data.copy()
mask_data = mask_data.copy()

# Directory paths for saving images and masks
os.makedirs('train_images', exist_ok=True)
os.makedirs('train_masks', exist_ok=True)

# Initialize lists to store metrics
psnr_list = []
ssim_list = []
cnr_list = []

# Iterate through each slice in the CT scan
for slice_idx in range(ct_data.shape[2]):
    ct_slice = ct_data[:, :, slice_idx]
    mask_slice = mask_data[:, :, slice_idx]

    # Skip empty slices (adjust condition based on your data characteristics)
    if np.any(ct_slice != 0):
        # Save CT slice as JPEG
        ct_tensor = tensor(ct_slice.astype(np.float32))
        ct_tensor.save_tiff_512x512(f"train_images/slice_{slice_idx}.tiff", [dicom_windows.liver, dicom_windows.custom])

        # Save mask slice as grayscale TIFF
        mask_tensor = tensor(mask_slice.astype(np.float32))
        mask_tensor.save_tiffgray_512x512(f"train_masks/slice_{slice_idx}.tiff", [dicom_windows.liver, dicom_windows.custom])

        # Calculate PSNR
        psnr_value = peak_signal_noise_ratio(ct_slice, mask_slice, data_range=ct_slice.max() - ct_slice.min())
        if not np.isnan(psnr_value):
            psnr_list.append(psnr_value)

        # Calculate SSIM
        ssim_value = structural_similarity(ct_slice, mask_slice, data_range=ct_slice.max() - ct_slice.min())
        if not np.isnan(ssim_value):
            ssim_list.append(ssim_value)

        # Calculate CNR
        cnr_value = calculate_cnr(ct_slice, mask_slice)
        if not np.isnan(cnr_value):
            cnr_list.append(cnr_value)

# Calculate average metrics
average_psnr = np.mean(psnr_list)
average_ssim = np.mean(ssim_list)
average_cnr = np.mean(cnr_list)

# Print average metrics
print(f"Average PSNR: {average_psnr:.2f}")
print(f"Average SSIM: {average_ssim:.2f}")
print(f"Average CNR: {average_cnr:.2f}")

# Print total number of slices processed
total_slices = len(psnr_list)
print(f"Total slices processed: {total_slices}")

# Save the metrics to a CSV file
metrics_dict = {
    'Average PSNR': [average_psnr],
    'Average SSIM': [average_ssim],
    'Average CNR': [average_cnr],
    'Total slices processed': [total_slices]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv('metrics_summary.csv', index=False)

# End of script
print('Processing completed.')
