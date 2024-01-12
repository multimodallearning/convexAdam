#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import numpy as np
import nibabel as nib
import struct
import scipy.ndimage
from scipy.ndimage import map_coordinates, zoom

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
import sys
import time
from scipy.ndimage import distance_transform_edt as edt


def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

    
import os
import pandas as pd
from scipy.ndimage.interpolation import zoom

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist
    
def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    # kernel size
    kernel_size = radius * 2 + 1
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6),indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    
    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean()*0.001, mind_var.mean()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind





#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);
    C = int(mind_fix.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
        mind_unfold = mind_unfold.view(C,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)
        

    ssd = torch.zeros((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    ssd_argmin = torch.zeros(H//grid_sp,W//grid_sp,D//grid_sp).long()
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).pow(2).sum(0,keepdim=True)
            #5,stride=1,padding=2
            #3,stride=1,padding=1
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#
        #ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    t1 = time.time()
    #print(t1-t0,'sec (ssd)')
    #gpu_usage()
    return ssd,ssd_argmin

#solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

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

def kpts_pt(kpts_world, shape):
    device = kpts_world.device
    H, W, D = shape
    return (kpts_world.flip(-1) / (torch.tensor([D, W, H]).to(device) - 1)) * 2 - 1

def kpts_world(kpts_pt, shape):
    device = kpts_pt.device
    H, W, D = shape
    return ((kpts_pt.flip(-1) + 1) / 2) * (torch.tensor([H, W, D]).to(device) - 1)

import math
import torch
import torch.nn.functional as F

class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n+4, n+4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.solve(v, A)[0]
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()
    
def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2**12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D//step, H//step, W//step
    
    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)
    
    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1*H1*W1
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)
        
    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
    
    return y2


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def combineDeformation3d_(disp_1st,disp_2nd,identity):
    disp_composition = disp_2nd + F.grid_sample(disp_1st.permute(0,4,1,2,3),disp_2nd+identity).permute(0,2,3,4,1)
    return disp_composition


# In[3]:


def find_rigid_3d(x, y):
    x_mean = x[:, :3].mean(0)
    y_mean = y[:, :3].mean(0)
    u, s, v = torch.svd(torch.matmul((x[:, :3]-x_mean).t(), (y[:, :3]-y_mean)))
    m = torch.eye(v.shape[0], v.shape[0]).to(x.device)
    m[-1,-1] = torch.det(torch.matmul(v, u.t()))
    rotation = torch.matmul(torch.matmul(v, m), u.t())
    translation = y_mean - torch.matmul(rotation, x_mean)
    T = torch.eye(4).to(x.device)
    T[:3,:3] = rotation
    T[:3, 3] = translation
    return T
def least_trimmed_rigid(fixed_pts, moving_pts, iter=5):
    idx = torch.arange(fixed_pts.shape[0]).to(fixed_pts.device)
    for i in range(iter):
        x = find_rigid_3d(fixed_pts[idx,:], moving_pts[idx,:]).t()
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x), 2), 1))
        _, idx = torch.topk(residual, fixed_pts.shape[0]//2, largest=False)
    return x.t()

def least_trimmed_squares(fixed_pts,moving_pts,iter=5):
    idx = torch.arange(fixed_pts.size(0)).to(fixed_pts.device)
    for i in range(iter):
        x,_ = torch.solve(moving_pts[idx,:].t().mm(moving_pts[idx,:]),moving_pts[idx,:].t().mm(fixed_pts[idx,:]))
        residual = torch.sqrt(torch.sum(torch.pow(moving_pts - torch.mm(fixed_pts, x),2),1))
        _,idx = torch.topk(residual,fixed_pts.size(0)//2,largest=False)
    return x


# In[4]:


#import matplotlib.pyplot as plt

grid_sp = 6#5
disp_hw = 6#7

TRE0_all = torch.zeros(22)
TRE_def_all = torch.zeros(22)
TRE_rigid_all = torch.zeros(22)
TRE_adam_all = torch.zeros(22)

R_all = torch.zeros(22,4,4)

H=W=256; D=288
for ii,nu in enumerate((1,2,3,4,5,6,7,8,12,13,14,15,16,17,18,19,21,23,24,25,26,27)):#1,12,25)):#

    img_moving = torch.from_numpy(nib.load('L2R_Task01/EASY-RESECT/NIFTI/Case'+str(nu)+'/Case'+str(nu)+'-T1-resize.nii').get_fdata()).float()
    img_moving2 = torch.from_numpy(nib.load('L2R_Task01/EASY-RESECT/NIFTI/Case'+str(nu)+'/Case'+str(nu)+'-FLAIR-resize.nii').get_fdata()).float()
    img_fixed = torch.from_numpy(nib.load('L2R_Task01/EASY-RESECT/NIFTI/Case'+str(nu)+'/Case'+str(nu)+'-US-before-resize.nii').get_fdata()).float()
    #print(img_fixed.shape)
    seg_moving = torch.from_numpy(nib.load('L2R_Task01/Case'+str(nu)+'-MRI-landmarks.nii.gz').get_fdata()).short().reshape(-1).cuda()
    seg_fixed = torch.from_numpy(nib.load('L2R_Task01/Case'+str(nu)+'-US-landmarks.nii.gz').get_fdata()).short().reshape(-1).cuda()
    mesh = torch.stack(torch.meshgrid((torch.arange(256),torch.arange(256),torch.arange(288)))).reshape(3,-1).float().cuda()
    coord_fixed = torch.empty(0,3)
    coord_moving = torch.empty(0,3)
    affine = F.affine_grid(torch.eye(3,4).cuda().unsqueeze(0),(1,1,H,W,D),align_corners=False)



    for i in range(1,int(seg_moving.max())+1):
        idx = torch.nonzero(seg_fixed==i)
        coord_fixed = torch.cat((coord_fixed,mesh[:,idx].mean(1).view(1,3).cpu()))
        idx = torch.nonzero(seg_moving==i)
        coord_moving = torch.cat((coord_moving,mesh[:,idx].mean(1).view(1,3).cpu()))

    TRE0 = (coord_fixed-coord_moving).pow(2).sum(1).sqrt()
    TRE0_all[ii] = TRE0.mean()
    #print('TRE0',TRE0,TRE0.mean())
    #compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        mindssc_fix = MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).cuda(),3,3).half()#[:,:,::2,::2,::2]#*fixed_mask.cuda().half()#.cpu()
        mindssc_mov = MINDSSC(img_moving.unsqueeze(0).unsqueeze(0).cuda(),3,3).half()#[:,:,::2,::2,::2]#*moving_mask.cuda().half()#.cpu()
        mindssc_mov2 = MINDSSC(img_moving2.unsqueeze(0).unsqueeze(0).cuda(),3,3).half()#[:,:,::2,::2,::2]#*moving_mask.cuda().half()#.cpu()
        mind_fix = torch.cat((F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp),F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)),1)
        mind_mov = torch.cat((F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp),F.avg_pool3d(mindssc_mov2,grid_sp,stride=grid_sp)),1)
        mask_mov = F.avg_pool3d((img_moving>10).cuda().float().unsqueeze(0).unsqueeze(0),grid_sp,stride=grid_sp)>.5

        mask_fix = F.avg_pool3d((img_fixed>10).cuda().float().unsqueeze(0).unsqueeze(0),grid_sp,stride=grid_sp)>.5

        scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2


        ssd,ssd_argmin = correlate(mind_fix,mind_mov,disp_hw,grid_sp,(H,W,D))
        ssd *= mask_fix.squeeze(1)
        disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)
        disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))
        disp_hr = F.interpolate(disp_soft*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
        disp0 = disp_hr.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
        disp0 = disp0.flip(4)
        
        
        #print(R)
        #R_all[ii] = R.cpu()
        
        del ssd
        ssd_,ssd_argmin_ = correlate(mind_mov,mind_fix,disp_hw,grid_sp,(H,W,D))
        ssd_ *= mask_mov.squeeze(1)
        disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
        disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=5)
        del ssd_
        torch.cuda.empty_cache()
        disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
        #t_convexmind += time.time()-t0
        disp0 = disp_hr.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
        disp0 = disp0.flip(4)
        
        affine_sp = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
        affine_sp = affine_sp.reshape(-1,3)[torch.nonzero(mask_fix.reshape(-1)),:]

        T1 = F.grid_sample(affine.permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
        T2 = F.grid_sample((affine+disp0).permute(0,4,1,2,3),affine_sp.reshape(1,-1,1,1,3))
        T1 = torch.cat((T1.squeeze().t(),torch.ones(affine_sp.shape[0],1).cuda()),1)
        T2 = torch.cat((T2.squeeze().t(),torch.ones(affine_sp.shape[0],1).cuda()),1)
        
        R = least_trimmed_rigid(T1,T2,15)#torch.cat((T1,T1_),0),torch.cat((T2,T2_),0))


        affineR = F.affine_grid(R[:3].unsqueeze(0),(1,1,H,W,D),align_corners=False)
        
        

        

        warped_seg = F.grid_sample(seg_moving.view(1,1,H,W,D).float().cuda(),affine+disp0,align_corners=False,mode='nearest')
        coord_warped = torch.empty(0,3)
        for i in range(1,int(seg_moving.max())+1):
            idx = torch.nonzero(warped_seg.reshape(-1)==i)
            coord_warped = torch.cat((coord_warped,mesh[:,idx].mean(1).view(1,3).cpu()))
        TRE_def = (coord_fixed-coord_warped).pow(2).sum(1).sqrt()
        TRE_def_all[ii] = TRE_def.mean()


        warped_seg = F.grid_sample(seg_moving.view(1,1,H,W,D).float().cuda(),affineR,align_corners=False,mode='nearest')
        coord_warped = torch.empty(0,3)
        for i in range(1,int(seg_moving.max())+1):
            idx = torch.nonzero(warped_seg.reshape(-1)==i)
            coord_warped = torch.cat((coord_warped,mesh[:,idx].mean(1).view(1,3).cpu()))
        TRE_rigid = (coord_fixed-coord_warped).pow(2).sum(1).sqrt()
        TRE_rigid_all[ii] = TRE_rigid.mean()
        print('Case',nu,'TRE (before,deformable,rigid)',TRE0.mean(),TRE_def.mean(),TRE_rigid.mean())
        #TRE_adam_all
        #TRE_adam_all[ii] = TRE_adam.mean()

        gpu_usage()




# In[6]:


#plt.plot(TRE0_all)
#plt.plot(TRE_def_all)
#plt.plot(TRE_rigid_all)
#plt.show()
print(TRE0_all.mean()*.5,TRE_def_all.mean()*.5,TRE_rigid_all.mean()*.5,)

