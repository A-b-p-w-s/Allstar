import torch
import torch.optim as optim
import nibabel as nib
from torch.utils.data import DataLoader
import model.attention_unet as attention_unet
from datasets.dataset import CT_CTA_Dataset, collate_fn_nii, map_data_2, unmap_data_2
# from attention_unet import AttResUNet, Discriminator
from model.unet_cbam import UNet_CBAM, Discriminator2, Discriminator
# from gan import Generator, Discriminator
from model.transformer_2d import Transformer_2D
# from reg import Reg
from criterion import (discriminator_loss, generator_loss,
                       gradient_penalty, smoothing_loss, ssim_metric, psnr, mse)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, here must should be 1')
    parser.add_argument('--num_slice', type=int, default=16, help='num of slice per iteration')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_drop', type=int, default=8, help='lr drop step')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs')
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
    
    # coefficients
    parser.add_argument('--coef_g', type=float, default=1, help='coef of smoothing loss')
    parser.add_argument('--coef_l1', type=float, default=0.8, help='coef of mae loss')
     
    # trian params
    parser.add_argument('--resume', action='store_true', help='resume')
    parser.add_argument('--val', action='store_true', help='val')
    parser.add_argument('--model_path', type=str, default=r'./output/model_10.pth', help='model path')
    parser.add_argument('--output',default=r'./output/', help='output dir')
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

def define_optimizers(generator, registor, discriminator, learning_rate):
    # optimizer_G = optim.SGD([{'params':generator.parameters(), 'lr':learning_rate},
    #                         {'params':registor.parameters(), 'lr':learning_rate}],
    #                         weight_decay=0.0001)
    
    # optimizer_G = optim.RMSprop([{'params':generator.parameters(), 'lr':learning_rate},
    #                         {'params':registor.parameters(), 'lr':learning_rate}],
    #                         weight_decay=0.0001)
    optimizer_G = optim.Adam([{'params':generator.parameters(), 'lr':learning_rate/2},
                            {'params':registor.parameters(), 'lr':learning_rate}],
                            weight_decay=0.0001, betas=(0.5,0.999))

    # optimizer_D = optim.SGD(discriminator.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer_D = optim.RMSprop(discriminator.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=0.0001, betas=(0.5,0.999))
    return optimizer_G, optimizer_D

def train(args, generator, registor, discriminator, train_dataloader, val_dataloader,
          device=None, logger=None):
    # Define optimizers
    optimizer_G, optimizer_D = define_optimizers(generator, registor, discriminator, learning_rate=args.lr)
    lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_G, args.lr_drop, gamma=0.1)
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_D, args.lr_drop, gamma=0.1)

    # resume
    if args.resume:
        print('loading pretrained weights')
        load_checkpoint(args.model_path, generator, discriminator, registor, optimizer_G, optimizer_D)
    
    # 重采样函数
    # transformer = Transformer_2D().to(device)
    # regist_loss = GradientConsistencyLoss(kernel_type='sobel')
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch:{epoch+1}:optimizer_G learning rate:{optimizer_G.param_groups[0]['lr']}  optimizer_D learning rate:{optimizer_D.param_groups[0]['lr']}")
        generator = generator.to(device).train()
        discriminator = discriminator.to(device).train()
        # registor = registor.to(device).train()

        writer = SummaryWriter(log_dir=args.output + '/logs')
        Dis_loss = []
        Gen_loss = []
        Reg_loss = []
        MAE_loss = []
        PSNR = []
        
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}|  {elapsed}|\t', ncols=70)
        loop.set_description(f'Epoch {epoch+1}/{args.epochs}')
        for i, (ct_data, cta_data) in loop:
            ct_img = ct_data['img'].squeeze(0)
            cta_img = cta_data['img'].squeeze(0)
            
            split_ct_img = ct_img.split(args.num_slice, dim=0)
            split_cta_img = cta_img.split(args.num_slice, dim=0)
    
            for ct, cta in zip(split_ct_img,split_cta_img):
                ct, cta = ct.to(device), cta.to(device)
                
                # Update discriminator: maximize D(x) + (1 - D(G(x)))
                for n in range(3):
                    optimizer_D.zero_grad()
                    with torch.no_grad():
                        # flow0 = registor(ct).detach()
                        # reg_ct0 = transformer(ct, flow0).detach()
                        fake_cta = generator(ct).detach()
                    real_logits = discriminator(cta)
                    fake_logits = discriminator(fake_cta)
                    gradient_penalty_loss = gradient_penalty(discriminator, cta, fake_cta, device=device)
                    d_loss = discriminator_loss(real_logits, fake_logits)
                    Dis_loss.append(d_loss.item())
                    d_gp_loss = d_loss + 10*gradient_penalty_loss
                    d_gp_loss.backward()
                    optimizer_D.step()
                   
                
                # Update generator: minimize D(G(x))
                optimizer_G.zero_grad()
                # flow = registor(ct)
                # reg_ct = transformer(ct, flow)
                # r_loss = regist_loss(reg_ct, cta)
                # Reg_loss.append(r_loss.item())

                fake_cta = generator(ct) 
                fake_logits = discriminator(fake_cta)
                g_loss = generator_loss(fake_logits)
                Gen_loss.append(g_loss.item())
                mae_loss = F.l1_loss(fake_cta, cta)
                MAE_loss.append(mae_loss.item())
                
                total_g_loss = g_loss + args.coef_l1 * mae_loss
                total_g_loss.backward()
                optimizer_G.step()
            
                # Print log info
                # MAE += mae(reg_fake_cta,cta).item()
                PSNR.append(psnr(fake_cta,cta).item())
                
            if (i) % 5 == 0:

                # logger.info(f'D_Loss: {np.mean(Dis_loss):.4f}  G_Loss: {np.mean(Gen_loss):.4f}  R_Loss: {np.mean(Reg_loss):.4f}  PSNR:{np.mean(PSNR):.4f}')
                logger.info(f'D_Loss: {np.mean(Dis_loss):.4f}  G_Loss: {np.mean(Gen_loss):.4f}  MAE_Loss: {np.mean(MAE_loss):.4f}  PSNR:{np.mean(PSNR):.4f}')
               
       
        print(f"epoch:{epoch+1}, Dis_loss: {np.mean(Dis_loss)}, Gen_loss: {np.mean(Gen_loss)}, MAE_Loss: {np.mean(MAE_loss):.4f}")
        writer.add_scalar('Dis_loss', np.mean(Dis_loss), epoch)
        writer.add_scalar('Gen_loss', np.mean(Gen_loss), epoch)
        # writer.add_scalar('Reg_Mae', np.mean(Reg_loss), epoch)
        writer.add_scalar('MAE', np.mean(MAE_loss), epoch)
        writer.add_scalar('PSNR', np.mean(PSNR), epoch)
        
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        
        # torch.cuda.empty_cache()
        
        # 保存每个epoch的模型
        if not os.path.exists('./output'):
            os.makedirs('./output')
        save_checkpoint({
            'epoch': epoch + 1,
            'g_dict': generator.state_dict(),
            'd_dict': discriminator.state_dict(),
            'c_dict': registor.state_dict(),
            'optimizer_g': optimizer_G.state_dict(),
            'optimizer_d': optimizer_D.state_dict(),
            }, filename=f'./output/model_{epoch+1}.pth')
        
        
        val(args, generator,registor, val_dataloader, device)
        
        # torch.cuda.empty_cache()
        writer.close()

def val(args, generator, registor, dataloader, device):
    """
    Save the generated images to the output folder
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    generator = generator.to(device)
    # registor = registor.to(device)
    # transformer = Transformer_2D().to(device)
    generator.eval()
    # registor.eval()
    SSIM = []
    loop = tqdm(enumerate(dataloader), total=len(dataloader),
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}|  {elapsed}|\t', ncols=70)
    loop.set_description(f'Val')
    for i, (ct_data, cta_data) in loop:
            ct_img, cta_img = ct_data['img'].squeeze(0), cta_data['img'].squeeze(0)
            gen_cta = torch.tensor([],requires_grad=False).to(device)
            split_ct_img = ct_img.split(args.num_slice, dim=0) 
            for ct in split_ct_img:
                ct =ct.to(device)
                with torch.no_grad():
                    # flow = registor(ct)
                    # reg_ct = transformer(ct, flow)
                    fake_cta = generator(ct) 
                gen_cta = torch.cat((gen_cta, fake_cta), dim=0)
            
            gen_cta = gen_cta.cpu()
            ssim = ssim_metric(cta_img, gen_cta)
            SSIM.append(ssim.item())
            # print(f'ssim:{ssim:.4f}')
            gen_cta = unmap_data_2(gen_cta)
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
    img_data = img_data.permute(1, 2, 3, 0).squeeze(0)
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
    if registor is not None:
        registor.load_state_dict(state['c_dict'])
    # if optimizer_g is not None:
    #     optimizer_g.load_state_dict(state['optimizer_g'])
    # if optimizer_d is not None:
    #     optimizer_d.load_state_dict(state['optimizer_d'])
    # if optimizer_c is not None:
    #     optimizer_c.load_state_dict(state['optimizer_c'])
    args.start_epoch = state['epoch']
   
    return


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
    
    generator = UNet_CBAM(in_channels=1, out_channels=1)
    discriminator = Discriminator2(in_channels=1, out_channels=1)
    registor = torch.nn.Linear(1,1)
    num_para = sum(p.numel() for p in generator.parameters()) \
            + sum(p.numel() for p in discriminator.parameters()) \
            + sum(p.numel() for p in registor.parameters())
    print(f"number of model's parameters:{num_para}")
    
    
    train_dataset = CT_CTA_Dataset(ct_dir=args.val_ct_dir, cta_dir=args.val_cta_dir,
                                   transform=map_data_2, args=args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, collate_fn=collate_fn_nii)
    val_dataset = CT_CTA_Dataset(ct_dir=args.test_ct_dir, cta_dir=args.test_cta_dir,
                                    transform=map_data_2, args=args)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, collate_fn=collate_fn_nii) 
    

    if args.val:
        print("val mode")
        # resume
        if args.resume:
            print('loading pretrained weights')
            load_checkpoint(args.model_path, generator,discriminator,registor)
        val(args, generator, registor, val_dataloader, device)
    else:
        print("train mode")
        train(args=args, generator=generator, registor=registor, discriminator=discriminator, 
            train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device, logger=logger)


