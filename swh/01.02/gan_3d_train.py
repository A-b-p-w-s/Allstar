import torch
import torch.optim as optim
import nibabel as nib
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets.dataset import CT_CTA_Dataset, collate_fn_nii, unmap_data
from model.unet_3d import UNet3D, Discriminator3D
# from model.transformer_2d import Transformer_2D
# from model.reg import Reg
from criterions.criterion import (discriminator_loss, generator_loss, gradient_penalty, smoothing_loss, ssim_metric, psnr)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import logging
import argparse
import math


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, here must should be 1')
    parser.add_argument('--num_slice', type=int, default=9, help='num of slice per iteration')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate of generator')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate of discriminator')
    parser.add_argument('--lr_drop', type=int, default=5, help='lr drop step')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr drop rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--manual_seed', type=int, default=111, help='random seed')
     
    # dataset params
    parser.add_argument('--wc', type=float, default=0, help='window center')
    parser.add_argument('--ww', type=float, default=400, help='window width')
    parser.add_argument('--train_ct_dir', type=str, default=r'D:\data\CTA\train\A3', help='A3 is CT dir')
    parser.add_argument('--train_cta_dir', type=str, default=r'D:\data\CTA\train\A0', help='A0,A1,A2 is CTA dir')
    parser.add_argument('--val_ct_dir', type=str, default=r'D:\data\CTA\val\A3', help='A3 is CT dir')
    parser.add_argument('--val_cta_dir', type=str, default=r'D:\data\CTA\val\A0', help='A0,A1,A2 is CTA dir')
    parser.add_argument('--test_ct_dir', type=str, default=r'D:\data\CTA\test\A3', help='A3 is CT dir')
    parser.add_argument('--test_cta_dir', type=str, default=r'D:\data\CTA\test\A0', help='A0,A1,A2 is CTA dir')
    
    # coefficients
    parser.add_argument('--coef_g', type=float, default=1, help='coef of smoothing loss')
    parser.add_argument('--coef_l2', type=float, default=3, help='coef of mae loss')
    parser.add_argument('--coef_sm', type=float, default=0.5, help='coef of mae loss')
    parser.add_argument('--coef_gp', type=float, default=10, help='coef of mae loss')
     
    # trian params
    parser.add_argument('--resume', action='store_true', help='resume')
    parser.add_argument('--val', action='store_true', help='val')
    parser.add_argument('--model_path', type=str, default=r'./output/model_4.pth', help='model path')
    args = parser.parse_args()
    return args 

def get_logger(log_dir='./output/logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建日志记录器   
    logger = logging.getLogger("Train_Gan_log")
    logger.setLevel(logging.INFO)
    # 创建控制台处理器，输出日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件处理器，保存日志到文件
    # log_file = os.path.join(log_dir, 'training-' + time.strftime('%y-%m-%d', time.localtime()) + '.log')
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO) 
    # 创建格式化器
    formatter = logging.Formatter('%(message)s')
    # file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 添加处理器到 logger
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def train(args, generator, discriminator, train_dataloader, val_dataloader, device=None, logger=None):
    # Define optimizers
    optimizer_G = optim.Adam([{'params':generator.parameters(), 'lr':args.lr_g}], betas=(0.9,0.999))

    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5,0.9))
   
    frequency = 5
    lr_scheduler_g = scheduler(optimizer_G, warm_up_steps=300//frequency, total_steps=2000, base_lr=args.lr_g, min_lr=1e-6)
    lr_scheduler_d = scheduler(optimizer_D, warm_up_steps=300//frequency, total_steps=2000, base_lr=args.lr_d, min_lr=1e-6)

    # resume
    if args.resume:
        print('loading pretrained weights')
        load_checkpoint(args.model_path, generator=generator, discriminator=discriminator)
    
    # 重采样函数
    # transformer = Transformer_2D().to(device)
    # regist_loss = GradientConsistencyLoss(kernel_type='sobel')

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch:{epoch+1}:optimizer_G learning rate:{optimizer_G.param_groups[0]['lr']}  optimizer_D learning rate:{optimizer_D.param_groups[0]['lr']}")
        generator = generator.to(device).train()
        discriminator = discriminator.to(device).train()
        # registor = registor.to(device).train()
        
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}|  {elapsed}|\t', ncols=70)
        loop.set_description(f'Epoch {epoch+1}/{args.epochs}')
        for i, (ct_data, cta_data) in loop:
            # dloss, gloss, mseloss = [], [], []
            psnr_list = []
            ct_img = ct_data['img'].permute(0,3,1,2).unsqueeze(0)
            cta_img = cta_data['img'].permute(0,3,1,2).unsqueeze(0)
            
            split_ct_img = ct_img.split(args.num_slice, dim=2)
            split_cta_img = cta_img.split(args.num_slice, dim=2)
            # d_loss = 0
            if (i+1) % frequency != 0:
                for ct, cta in zip(split_ct_img, split_cta_img):
                    ct, cta = ct.to(device), cta.to(device)
                    # Update discriminator: maximize D(x) + (1 - D(G(x)))
                    optimizer_D.zero_grad()
                    # with torch.no_grad():
                    fake_cta = generator(ct).detach()
                    real_logits = discriminator(cta)
                    fake_logits = discriminator(fake_cta)
                    gp_loss = gradient_penalty(discriminator, cta, fake_cta, device=device)
                    d_loss = discriminator_loss(real_logits, fake_logits)
                    dloss = d_loss.item()
                    # dloss.append(d_loss.item())
                    d_gp_loss = d_loss + args.coef_gp * gp_loss
                    # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2)
                    d_gp_loss.backward()
                    optimizer_D.step()
            else:
                for ct, cta in zip(split_ct_img, split_cta_img):
                    ct, cta = ct.to(device), cta.to(device)
                    # Update generator: minimize D(G(x))
                    optimizer_G.zero_grad()
                    fake_cta = generator(ct) 
                    fake_logits = discriminator(fake_cta)
                    g_loss = generator_loss(fake_logits)
                    gloss = g_loss.item()

                    # gloss.append(g_loss.item())
                    # flow = registor(ct,cta)
                    # reg_fake_cta = transformer(fake_cta, flow)
                    # sm_loss = smoothing_loss(flow)
                    # # Reg_loss.append(sm_loss.item())
                    mse_loss = F.mse_loss(fake_cta, cta)
                    mseloss = mse_loss.item()
                    # mseloss.append(mse_loss.item())
                    
                    total_g_loss = args.coef_g * g_loss + args.coef_l2 * mse_loss #+ args.coef_sm * sm_loss
                    # total_g_loss = g_loss
                    total_g_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(generator.parameters(), 1, norm_type=2)
                    optimizer_G.step()
                    psnr_list.append(psnr(fake_cta,cta).item())

                   
            if (i+1) % frequency == 0: 
                psnr_m = np.mean(psnr_list)
                logger.info(f'D_Loss: {dloss:.4f}  G_Loss: {gloss:.4f}  MSE_Loss: {mseloss:.4f}  PSNR: {psnr_m:.4f}  LR: {lr_scheduler_g.get_last_lr()[0]:.4e}')
                lr_scheduler_g.step()
                lr_scheduler_d.step()
    
        # logger.info(f"epoch:{epoch+1}, Dis_loss: {np.mean(Dis_loss)}, Gen_loss: {np.mean(Gen_loss)}, MSE_Loss: {np.mean(MSE_loss):.4f}")
        
        
        # 保存每个epoch的模型
        if not os.path.exists('./output'):
            os.makedirs('./output')
        save_checkpoint({
            'epoch': epoch + 1,
            'g_dict': generator.state_dict(),
            'd_dict': discriminator.state_dict(),
            # 'r_dict': registor.state_dict(),
            # 'optimizer_g': optimizer_G.state_dict(),
            # 'optimizer_d': optimizer_D.state_dict(),
            }, filename=f'./output/model_{epoch+1}.pth')
        
        
        val(args, generator, val_dataloader, device)
        

def val(args, generator, dataloader=None, device=None):
    """
    Save the generated images to the output folder
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    generator = generator.to(device).eval()
    # generator.eval()
    # registor = registor.to(device).eval()
    # transformer = Transformer_2D().to(device)
    # registor.eval()
    SSIM = []
    loop = tqdm(enumerate(dataloader), total=len(dataloader),
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}|  {elapsed}|\t', ncols=70)
    loop.set_description(f'Val')
    for i, (ct_data, cta_data) in loop:
            ct_img, cta_img = ct_data['img'].to(device).permute(0,3,1,2).unsqueeze(0), cta_data['img'].to(device).permute(0,3,1,2).unsqueeze(0)

            gen_cta = torch.tensor([],requires_grad=False).to(device)
            split_ct_img = ct_img.split(args.num_slice, dim=2)
            split_cta_img = cta_img.split(args.num_slice, dim=2)

            for ct, cta in zip(split_ct_img, split_cta_img):
                # ct, cta = ct.to(device), cta.to(device)
                with torch.no_grad():
                    # flow = registor(ct,cta)
                    fake_cta = generator(ct) 
                    # reg_fake_cta = transformer(fake_cta, flow)
                gen_cta = torch.cat((gen_cta, fake_cta), dim=2)
            
            gen_cta = gen_cta.cpu()
            cta_img = cta_img.cpu()
            ssim = ssim_metric(cta_img, gen_cta)
            SSIM.append(ssim.item())
            # print(f'ssim:{ssim:.4f}')
            gen_cta = unmap_data(gen_cta)
            save_nifti_image(img_data=gen_cta, ori_affine=ct_data['affine'], 
                            ori_header=ct_data['header'], ori_path=ct_data['path'])
            
            if i == 4:
                break
    
    print(f'SSIM:{np.mean(SSIM):.4f}')
    return
                               
def save_nifti_image(img_data, ori_affine, ori_header, ori_path):
    if not os.path.exists('./output/generated'):
        os.makedirs('./output/generated')
    file_name = os.path.basename(ori_path)
    img_data = img_data.squeeze(0).squeeze(0).permute(1, 2, 0)
    new_img = nib.Nifti1Image(img_data.cpu().numpy(), affine=ori_affine, header=ori_header)
    nib.save(new_img, './output/generated/GEN-' + file_name)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(checkpoint_file, generator=None, discriminator=None, registor=None,
                    optimizer_g=None, optimizer_d=None):
    state = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    if generator is not None:
        generator.load_state_dict(state['g_dict'])
    if discriminator is not None:
        discriminator.load_state_dict(state['d_dict'])
    # if registor is not None:
    #     registor.load_state_dict(state['r_dict'])
    # if optimizer_g is not None:
    #     optimizer_g.load_state_dict(state['optimizer_g'])
    # if optimizer_d is not None:
    #     optimizer_d.load_state_dict(state['optimizer_d'])
    # if optimizer_c is not None:
    #     optimizer_c.load_state_dict(state['optimizer_c'])
    args.start_epoch = state['epoch']
   
    return

def scheduler(optimizer, warm_up_steps, total_steps, base_lr, min_lr):
    def lr_lambda(current_step):
        if current_step < warm_up_steps:
            # warm-up阶段，学习率线性增长
            return current_step / warm_up_steps
        else:
            # 余弦退火阶段
            cos_decay = 0.5 * (1 + math.cos(math.pi * (current_step - warm_up_steps) / (total_steps - warm_up_steps)))
            return (min_lr / base_lr) + (cos_decay * (1 - min_lr / base_lr))
        
    return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
    logger = get_logger()
    args = get_args()
    print(args)
    
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = UNet3D(in_channel=1, out_channel=1, num_groups=8)
    discriminator = Discriminator3D(in_channel=1, num_groups=8)
    # registor = Reg(512,512,1,1)
    # registor = torch.nn.Linear(1,1)
    num_para = sum(p.numel() for p in generator.parameters()) \
            + sum(p.numel() for p in discriminator.parameters()) 
            # + sum(p.numel() for p in registor.parameters())
    print(f"number of model's parameters:{num_para}")
    
    
    train_dataset = CT_CTA_Dataset(ct_dir=args.train_ct_dir, cta_dir=args.train_cta_dir, args=args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, collate_fn=collate_fn_nii)
    
    val_dataset = CT_CTA_Dataset(ct_dir=args.test_ct_dir, cta_dir=args.test_cta_dir, args=args)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, collate_fn=collate_fn_nii) 
    

    if args.val:
        print("val mode")
        # resume
        if args.resume:
            print('loading pretrained weights')
            load_checkpoint(args.model_path, generator)
        val(args, generator, val_dataloader, device)
    else:
        print("train mode")
        train(args=args, generator=generator, discriminator=discriminator, train_dataloader=train_dataloader, 
              val_dataloader=val_dataloader, device=device, logger=logger)


