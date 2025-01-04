import torch
import nibabel as nib
from torch.utils.data import DataLoader
from datasets.dataset import CT_CTA_Dataset, collate_fn_nii, unmap_data
from model.unet_cbam import UNet_CBAM
# from model.unet_grfb import UNet_GRFB
from criterions.criterion import ssim_metric
from tqdm import tqdm
import numpy as np
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, here must should be 1')
    parser.add_argument('--num_slice', type=int, default=10, help='num of slice per iteration')
    parser.add_argument('--manual_seed', type=int, default=111, help='random seed')
     
    # dataset params
    parser.add_argument('--wc', type=float, default=30, help='window center')
    parser.add_argument('--ww', type=float, default=400, help='window width')
    parser.add_argument('--train_ct_dir', type=str, default=r'D:\data\CTA\train\A3', help='A3 is CT dir')
    parser.add_argument('--train_cta_dir', type=str, default=r'D:\data\CTA\train\A0', help='A0,A1,A2 is CTA dir')
    parser.add_argument('--val_ct_dir', type=str, default=r'D:\data\CTA\val\A3', help='A3 is CT dir')
    parser.add_argument('--val_cta_dir', type=str, default=r'D:\data\CTA\val\A0', help='A0,A1,A2 is CTA dir')
    parser.add_argument('--test_ct_dir', type=str, default=r'D:\data\CTA\test\A3', help='A3 is CT dir')
    parser.add_argument('--test_cta_dir', type=str, default=r'D:\data\CTA\test\A0', help='A0,A1,A2 is CTA dir')
     
    # trian params
    parser.add_argument('--model_path', type=str, default=r'./output/model_24.pth', help='model path')
    parser.add_argument('--output', default=r'./output', help='output dir')
    args = parser.parse_args()
    return args 


def predict(args, generator, dataloader, device):
    """
    Save the generated images to the output folder
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    generator = generator.to(device).eval()

    loop = tqdm(enumerate(dataloader), total=len(dataloader),
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}|  {elapsed}|\t', ncols=70)
    loop.set_description(f'Val')
    for i, ct_data in loop:
            ct_img = ct_data['img'].squeeze(0)

            gen_cta = torch.tensor([],requires_grad=False).to(device)
            split_ct_img = ct_img.split(args.num_slice, dim=0)

            for ct in split_ct_img:
                ct = ct.to(device)
                with torch.no_grad():
                    fake_cta = generator(ct) 
                gen_cta = torch.cat((gen_cta, fake_cta), dim=0)
            
            gen_cta = gen_cta.cpu()
            gen_cta = unmap_data(gen_cta)
            save_nifti_image(img_data=gen_cta, ori_affine=ct_data['affine'], 
                            ori_header=ct_data['header'], ori_path=ct_data['path'])
            
            if i == 4:
                break
    print('process finished')
    return
                               
def save_nifti_image(img_data, ori_affine, ori_header, ori_path):
    if not os.path.exists(args.output + '/predict'):
        os.makedirs(args.output + '/predict')
    file_name = os.path.basename(ori_path)
    img_data = img_data.permute(1, 2, 3, 0).squeeze(0)
    new_img = nib.Nifti1Image(img_data.cpu().numpy(), affine=ori_affine, header=ori_header)
    nib.save(new_img, './output/predict/Pred-' + file_name)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(checkpoint_file, generator=None, discriminator=None, registor=None,
                    optimizer_g=None, optimizer_d=None):
    state = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    if generator is not None:
        generator.load_state_dict(state['g_dict'])
    if discriminator is not None:
        discriminator.load_state_dict(state['d_dict'])
    if registor is not None:
        registor.load_state_dict(state['r_dict'])
    # if optimizer_g is not None:
    #     optimizer_g.load_state_dict(state['optimizer_g'])
    # if optimizer_d is not None:
    #     optimizer_d.load_state_dict(state['optimizer_d'])
    # if optimizer_c is not None:
    #     optimizer_c.load_state_dict(state['optimizer_c'])
    args.start_epoch = state['epoch']
   
    return


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = UNet_CBAM(in_channels=1, out_channels=1)

    # registor = torch.nn.Linear(1,1)
    num_para = sum(p.numel() for p in generator.parameters())

    print(f"number of model's parameters:{num_para}")
    
    
    test_dataset = CT_CTA_Dataset(ct_dir=args.test_ct_dir, args=args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, 
                                 shuffle=False, num_workers=2, collate_fn=collate_fn_nii) 
    
 
    print('loading pretrained weights')
    load_checkpoint(args.model_path, generator)
    predict(args, generator, test_dataloader, device)



