'''
Copyright © 2024 泰玛 Tarek Ali, Allstar Medical. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from nilearn.image import resample_img
import skimage.transform as skTrans
import nibabel as nib
import cv2
import imageio
from tqdm.notebook import tqdm
from PIL import Image
import re
from fastai.basics import *
from fastai.vision import *
from fastai.data.transforms import *


print('processing images...')
###################################################
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

file_list = []

for dirname, _, filenames in os.walk('Task03_Liver/imagesTr'):
    for filename in filenames:
        file_list.append((dirname, filename))

df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])

# Apply natural sort to the filenames
df_files['filename'] = df_files['filename'].apply(str)
df_files = df_files.sort_values(by='filename', key=lambda x: x.map(natural_keys)).reset_index(drop=True)

########################################################

######################
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    # array = np.array(array)
    array = np.rot90(np.array(array))
    return array  # Return the raw numpy array


#######################################
# utils
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
def save_jpg(x: (Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    # 可选：水平翻转图像
    # im = im.transpose(Image.FLIP_LEFT_RIGHT)
    # 可选：垂直翻转图像
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.save(fn, quality=quality)

@patch
def save_jpg_256x256(x: (Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im = im.resize((256, 256))
    im = im.rotate(angle=270)
    im.save(fn, quality=quality)


## GENERATE

GENERATE_JPG_FILES = True
slice_sum = 0
if (GENERATE_JPG_FILES):
    path = Path(".")
    os.makedirs('train_images', exist_ok=True)
    # for ii in range(0, len(df_files)):
    for ii in range(0, 1):
        # curr_ct = read_nii(df_files.loc[ii, 'dirname'] + "/" + df_files.loc[ii, 'filename'])
        curr_ct = read_nii(r'D:\hhf\test\a.nii')

        # curr_file_name = str(df_files.loc[ii, 'filename']).split('.')[0]
        curr_dim = curr_ct.shape[2]  # Assuming CT scans are 3D volumes

        slice_sum = slice_sum + curr_dim

        for curr_slice in range(0, curr_dim, 1):  # Export every slice for training
            data = tensor(curr_ct[..., curr_slice].astype(np.float32))
            data.save_jpg(f"C:/Users/allstar/Desktop/enhance_test/img2/{curr_slice}.jpg",
                                  [dicom_windows.liver, dicom_windows.custom])
            # data.save_jpg(f"train_images/{curr_file_name}_slice_{curr_slice}.jpg",
            #                       [dicom_windows.liver, dicom_windows.custom])

else:
    path = Path("nanana")

print(slice_sum)
