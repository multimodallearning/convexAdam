import numpy as np
import nibabel as nib
import struct
import scipy.ndimage
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
import sys
import time


sys.path.append('voxelmorph/pytorch/')
import losses
print(losses.mind_loss)

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))


H = 160
W = 192
D = 224

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


identity = np.stack(np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij'))
#print(fixed.shape)
#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp):
    torch.cuda.synchronize()
    C_mind = mind_fix.shape[1]
    t0 = time.time()
    with torch.no_grad():
        mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
        mind_unfold = mind_unfold.view(C_mind,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)
        

    ssd = torch.zeros((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    ssd_argmin = torch.zeros(H//grid_sp,W//grid_sp,D//grid_sp).long()
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).abs().sum(0,keepdim=True)
            
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#
        #ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    t1 = time.time()
    print(t1-t0,'sec (ssd)')
    gpu_usage()
    return ssd,ssd_argmin

#solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp):
    disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)


    coeffs = torch.tensor([0.003,0.01,0.03,0.1,0.3,1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H//grid_sp):

                coupled = ssd[:,i,:,:]+coeffs[j]*(disp_mesh_t-disp_soft[:,:,i].view(3,1,-1)).pow(2).sum(0).view(-1,W//grid_sp,D//grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled,0)
            #print(coupled.shape)

        disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_coupled_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)

    return disp_soft

#enforce inverse consistency of forward and backward transform
def inverse_consistency(disp_field1s,disp_field2s,iter=20):
    #factor = 1
    B,C,H,W,D = disp_field1s.size()
    #make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3).to(disp_field1s.device).to(disp_field1s.dtype)
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i,disp_field2i

def combineDeformation3d(disp_1st,disp_2nd,identity):
    disp_composition = disp_2nd + F.grid_sample(disp_1st,disp_2nd.permute(0,2,3,4,1)+identity)
    return disp_composition

grid_sp = 2
disp_hw = 3

nu = 438

fixed = torch.from_numpy(nib.load('L2R2021/Task3/skull_stripped/nnunet/img0'+str(nu)+'.nii.gz').get_fdata()).float()
moving = torch.from_numpy(nib.load('L2R2021/Task3/skull_stripped/nnunet/img0'+str(nu+1)+'.nii.gz').get_fdata()).float()


weight = 1/(torch.bincount(fixed.long().reshape(-1))+torch.bincount(moving.long().reshape(-1))).float().pow(.3)
weight /= weight.mean()
print(weight)
case_time = torch.zeros(38)
torch.cuda.synchronize()
t0_ = time.time()
for nu in range(1,39):
    torch.cuda.synchronize()
    t0 = time.time()
    fixed = torch.from_numpy(nib.load('/data_supergrover2/heinrich/nnUNet_predict/L2R_2021_Task3_test/img'+str(nu).zfill(4)+'_0000.nii.gz').get_fdata()).float()
    #    mindssc_fix_ = 10*(F.one_hot(fixed.cuda().view(1,H,W,D).long()).float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
    #RuntimeError: The size of tensor a (2) must match the size of tensor b (36) at non-singleton dimension 1

    moving = torch.from_numpy(nib.load('/data_supergrover2/heinrich/nnUNet_predict/L2R_2021_Task3_test/img'+str(nu+1).zfill(4)+'_0000.nii.gz').get_fdata()).float()

#/share/data_supergrover1/hansen/temp/nnUNet/nnUNet_results/nnUNet/3d_fullres/Task509_OASIS
#OASIS001_img.nii.gz
#/data_supergrover2/heinrich/nnUNet_predict/L2R_Task3_Test/nnunet/img
    fixed_seg = torch.from_numpy(nib.load('L2R2021/Task3/Lasse_nnUNet/img'+str(nu).zfill(4)+'.nii.gz').get_fdata()).float()
    moving_seg = torch.from_numpy(nib.load('L2R2021/Task3/Lasse_nnUNet/img'+str(nu+1).zfill(4)+'.nii.gz').get_fdata()).float()
    #moving_seg = torch.from_numpy(nib.load('/data_supergrover2/heinrich/nnUNet_predict/L2R_Task3_Test/nnunet/img'+str(nu+1).zfill(4)+'.nii.gz').get_fdata()).float()
    with torch.no_grad():
        mindssc_fix_ = 10*(F.one_hot(fixed_seg.cuda().view(1,H,W,D).long()).float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
        mindssc_mov_ = 10*(F.one_hot(moving_seg.cuda().view(1,H,W,D).long()).float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
        mind_fix_ = F.avg_pool3d(mindssc_fix_,grid_sp,stride=grid_sp)
        mind_mov_ = F.avg_pool3d(mindssc_mov_,grid_sp,stride=grid_sp)
        ssd,ssd_argmin = correlate(mind_fix_,mind_mov_,disp_hw,grid_sp)
        disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)
        disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp)

    del ssd,mind_fix_,mind_mov_


    print(disp_soft.shape)
    #del ssd
    #del disp_mesh_t
    torch.cuda.empty_cache()
    gpu_usage()



    disp_lr = F.interpolate(disp_soft*grid_sp,size=(H//2,W//2,D//2),mode='trilinear',align_corners=False)
#disp_soft*grid_sp/2#



    grid_sp = 2


    #extract one-hot patches
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        mind_fix_ = F.avg_pool3d(mindssc_fix_,grid_sp,stride=grid_sp)
        mind_mov_ = F.avg_pool3d(mindssc_mov_,grid_sp,stride=grid_sp)
    del mindssc_fix_,mindssc_mov_


    #extract one-hot patches
    
    #create optimisable displacement grid
    net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
    net[0].weight.data[:] = disp_lr/grid_sp
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)

    #run Adam optimisation with diffusion regularisation and B-spline smoothing
    lambda_weight = 1.25# sad: 10, ssd:0.75
    for iter in range(100):
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
        reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
        lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
        lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

        scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
        grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

        patch_mov_sampled = F.grid_sample(mind_mov_.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=False,mode='bilinear')#,padding_mode='border')
        #patch_mov_sampled_sq = F.grid_sample(mind_mov_.pow(2).float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=True,mode='bilinear')
        #sampled_cost = (patch_mov_sampled_sq-2*patch_mov_sampled*mind_fix_+mind_fix_.pow(2)).mean(1)*12
        sampled_cost = (patch_mov_sampled-mind_fix_).pow(2).mean(1)*12


        loss = sampled_cost.mean()
        (loss+reg_loss).backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    print(t1-t0,'sec (optim)')

    fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
    disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
    disp_field = F.interpolate(disp_hr,scale_factor = 0.5,mode='trilinear',align_corners=False)
    x1 = disp_field[0,0,:,:,:].cpu().float().data.numpy()
    y1 = disp_field[0,1,:,:,:].cpu().float().data.numpy()
    z1 = disp_field[0,2,:,:,:].cpu().float().data.numpy()

    #x1 = zoom(x,1/2,order=2).astype('float16')
    #y1 = zoom(y,1/2,order=2).astype('float16')
    #z1 = zoom(z,1/2,order=2).astype('float16')


    np.savez_compressed('/data_supergrover2/heinrich/L2R2021/convexAdam/submission/task_03/disp_'+str(nu).zfill(4)+'_'+str(nu+1).zfill(4)+'.npz',np.stack((x1,y1,z1),0))

    torch.cuda.synchronize()
    t1 = time.time()
    case_time[nu-1] = t1-t0

torch.cuda.synchronize()
t1_ = time.time()
print('total time',t1_-t0_)
torch.save(case_time,'/data_supergrover2/heinrich/L2R2021/convexAdam/task3_times.pth')
