import torch
import torch.optim as optim
import nibabel as nib
from torch.utils.data import DataLoader
from datasets.dataset import CT_CTA_Dataset, collate_fn_nii, unmap_data_2, map_data_2
from model.unet_cbam import UNet_CBAM
from model.transformer_2d import Transformer_2D
from criterion import (mse, ssim_metric, smooothing_loss, compute_edge)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os
import logging
import argparse

def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size, here should must be 1')
    parser.add_argument('--num_slice', type=int, default=16,
                        help='num of slice per iteration')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_drop', type=int, default=10, help='lr drop step')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--manual_seed', type=int, default=1, help='random seed')
    # parser.add_argument('--n_residual', type=int, default=3, help='num of residual blocks')
    
    # coefficients
    parser.add_argument('--coef_r', type=float, default=1, help='coef of smoothing loss')
    parser.add_argument('--coef_sm', type=float, default=5, help='coef of mae loss')
     
    # dataset params
    parser.add_argument('--wc', type=float, default=30, help='window center')
    parser.add_argument('--ww', type=float, default=400, help='window width')
    parser.add_argument('--train_ct_dir', type=str, default=r'D:\data\CTA\train\A3', 
                        help='A3 is CT dir')
    parser.add_argument('--train_cta_dir', type=str, default=r'D:\data\CTA\train\A0', 
                        help='A0,A1,A2 is CTA dir')
    parser.add_argument('--val_ct_dir', type=str, default=r'D:\data\CTA\val\A3', 
                        help='A3 is CT dir')
    parser.add_argument('--val_cta_dir', type=str, default=r'D:\data\CTA\val\A0',
                        help='A0,A1,A2 is CTA dir')
    parser.add_argument('--test_ct_dir', type=str, default=r'D:\data\CTA\test\A3', 
                        help='A3 is CT dir')
    parser.add_argument('--test_cta_dir', type=str, default=r'D:\data\CTA\test\A0', 
                        help='A0,A1,A2 is CTA dir')
    
    # trian params
    parser.add_argument('--resume', action='store_true', help='resume')
    parser.add_argument('--val', action='store_true', help='val')
    parser.add_argument('--rmodel_path', type=str, default=r'./output/reg_model_5.pth', 
                        help='model path')
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

def define_optimizers(registor, learning_rate=0.0001):
    optimizer = optim.Adam([{'params':registor.parameters(), 'lr':learning_rate}], weight_decay=0.001)
    return optimizer

def train(args, registor, train_dataloader, val_dataloader, device=None, logger=None):
    # Define optimizers
    optimizer = define_optimizers(registor, learning_rate=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.1)
    # resume
    if args.resume:
        print('loading pretrained weights')
        load_checkpoint(args.rmodel_path, registor=registor, optimizer=optimizer)
    
    transformer = Transformer_2D()
    # regist_loss = GradientConsistencyLoss(kernel_type='shcarr')
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch:{epoch+1}:optimizer learning rate:{optimizer.param_groups[0]['lr']}")
        registor = registor.to(device).train()
        SM_loss = []
        Reg_loss = []
        PSRN = []
        writer = SummaryWriter(log_dir=args.output + '/logs')

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
                optimizer.zero_grad()
                ct_edge = compute_edge(ct)
                flow = registor(ct_edge)
                reg_ct_edge = transformer(ct_edge, flow)
                cta_edge = compute_edge(cta)
                r_loss = mse(reg_ct_edge, cta_edge)
                Reg_loss.append(r_loss.item())
                sm_loss = smooothing_loss(flow)
                SM_loss.append(sm_loss.item())
                
                totalloss = args.coef_r * r_loss + args.coef_sm * sm_loss
                totalloss.backward()
                optimizer.step()
            
            # Print log info
            if (i) % 50 == 0:
                logger.info(f'Reg_loss: {np.mean(Reg_loss):.4f}, SM_loss: {np.mean(SM_loss):.4f}')
        

        print(f"epoch:{epoch+1}, Reg_loss:{np.mean(Reg_loss):.4f}")
        writer.add_scalar('Reg_loss', np.mean(Reg_loss), epoch)
        writer.add_scalar('SM_loss', np.mean(SM_loss), epoch)
        
        lr_scheduler.step()
        
        
        # 保存每个epoch的模型
        if not os.path.exists('./output'):
            os.makedirs('./output')
        save_checkpoint({
            'epoch': epoch + 1,
            # 'PN_dict': nmodel.state_dict(),
            'Reg_dict': registor.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, filename=f'./output/reg_model_{epoch+1}.pth')
        
        
        val(args, registor, val_dataloader, device)
        
          
        writer.close()

def val(args, registor, dataloader, device):
    """
    Save the generated images to the output folder
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    # resume
    if args.resume:
        print('loading pretrained weights')
        load_checkpoint(args.rmodel_path, registor=registor)
    

    registor = registor.to(device)
    registor.eval()
    transformer = Transformer_2D()
    for i, (ct_data, cta_data) in tqdm(enumerate(dataloader)):
            ct_img, cta_img = ct_data['img'].squeeze(0), cta_data['img'].squeeze(0)
            gen_cta = torch.tensor([],requires_grad=False).to(device)
            split_ct_img = ct_img.split(args.num_slice, dim=0) 
            for ct in split_ct_img:
                ct =ct.to(device)
                with torch.no_grad():
                    flow = registor(ct)
                    reg_ct = transformer(ct, flow)
                    gen_cta = torch.cat((gen_cta, reg_ct), dim=0)
            gen_cta = gen_cta.cpu()
            ssim = ssim_metric(cta_img, gen_cta.cpu())
            gen_cta = unmap_data_2(gen_cta).cpu()
            save_nifti_image(img_data=gen_cta, ori_affine=ct_data['affine'], 
                            ori_header=ct_data['header'], ori_path=ct_data['path'])
            print(f'ssim:{ssim:.4f}')
            if i ==4:
                break
    return
                               
def save_nifti_image(img_data, ori_affine, ori_header, ori_path):
    if not os.path.exists('./output/generated'):
        os.makedirs('./output/generated')
    file_name = os.path.basename(ori_path)
    img_data = img_data.permute(1, 2, 3, 0).squeeze(0)
    new_img = nib.Nifti1Image(img_data.cpu().numpy(), affine=ori_affine, header=ori_header)
    nib.save(new_img, './output/generated/Reg-' + file_name)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(checkpoint_file, model=None, registor=None, optimizer=None):
    state = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    if model is not None:
        model.load_state_dict(state['PN_dict'])
    if registor is not None:
        registor.load_state_dict(state['Reg_dict'])
        args.start_epoch = state['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
   
    return


if __name__ == '__main__':
    logger = get_logger()
    args = get_args()
    print(args)
    
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device:{device}')
    
    # nmodel = UNet_CBAM(in_channels=1, out_channels=1)
    rmodel = UNet_CBAM(in_channels=1, out_channels=2, flow=True)
    num_para = sum(p.numel() for p in rmodel.parameters())
    print(f"number of model's parameters:{num_para}")
    
    train_dataset = CT_CTA_Dataset(ct_dir=args.train_ct_dir, cta_dir=args.train_cta_dir,
                                   transform=map_data_2, args=args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=2, collate_fn=collate_fn_nii)
    val_dataset = CT_CTA_Dataset(ct_dir=args.test_ct_dir, cta_dir=args.test_cta_dir,
                                    transform=map_data_2, args=args)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn_nii) 
    

    if args.val:
        print("val mode")
        val(args, rmodel, val_dataloader, device)
    else:
        print("train mode")
        train(args=args, registor=rmodel, train_dataloader=train_dataloader, 
              val_dataloader=val_dataloader, device=device, logger=logger)


