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
from scipy.ndimage import distance_transform_edt as edt



sys.path.append('voxelmorph/pytorch/')
import losses
print(losses.mind_loss)

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))



H = 192
W = 192
D = 208

A = torch.ones(32,32).cuda()
A.requires_grad = True
A.sum().backward()

def load_case(nu):
    fixed = torch.from_numpy(nib.load('/data_supergrover2/heinrich/L2R_Lung/scans/case_0'+str(nu).zfill(2)+'_exp.nii.gz').get_fdata()).float()
    moving = torch.from_numpy(nib.load('/data_supergrover2/heinrich/L2R_Lung/scans/case_0'+str(nu).zfill(2)+'_insp.nii.gz').get_fdata()).float()


    fixed_mask = torch.from_numpy(nib.load('/data_supergrover2/heinrich/L2R_Lung/lungMasks/case_0'+str(nu).zfill(2)+'_exp.nii.gz').get_fdata()).float()
    moving_mask = torch.from_numpy(nib.load('/data_supergrover2/heinrich/L2R_Lung/lungMasks/case_0'+str(nu).zfill(2)+'_insp.nii.gz').get_fdata()).float()
    return fixed,moving,fixed_mask,moving_mask

#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        mind_unfold = F.unfold(F.pad(mind_mov,(disp_hw,disp_hw,disp_hw,disp_hw,disp_hw,disp_hw)).squeeze(0),disp_hw*2+1)
        mind_unfold = mind_unfold.view(12,-1,(disp_hw*2+1)**2,W//grid_sp,D//grid_sp)
        

    ssd = torch.zeros((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp,dtype=mind_fix.dtype, device=mind_fix.device)#.cuda().half()
    ssd_argmin = torch.zeros(H//grid_sp,W//grid_sp,D//grid_sp).long()
    with torch.no_grad():
        for i in range(disp_hw*2+1):
            mind_sum = (mind_fix.permute(1,2,0,3,4)-mind_unfold[:,i:i+H//grid_sp]).pow(2).sum(0,keepdim=True)
            #5,stride=1,padding=2
            #3,stride=1,padding=1
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


time_all = torch.zeros(10)


A = torch.ones(128,128).cuda()
A.requires_grad = True
A.sum().backward()
torch.cuda.synchronize()


for nu in range(21,31):
    torch.cuda.synchronize()
    t0 = time.time()
    fixed,moving,fixed_mask,moving_mask = load_case(nu)

    torch.cuda.synchronize()
    t0a = time.time()
    grid_sp = 4#4
    disp_hw = 6#6

    
    #replicate masking!!!
    avg3 = nn.Sequential(nn.ReplicationPad3d(1),nn.AvgPool3d(3,stride=1))
    avg3.cuda()
    mask = (avg3(fixed_mask.view(1,1,H,W,D).cuda())>0.9).float()
    dist,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
    fixed_r = F.interpolate((fixed[::2,::2,::2].cuda().reshape(-1)[idx[0]*104*96+idx[1]*104+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
    fixed_r.view(-1)[mask.view(-1)!=0] = fixed.cuda().reshape(-1)[mask.view(-1)!=0]
    #fixed_r = fixed.cuda().view(1,1,H,W,D)*mask
    mask = (avg3(moving_mask.view(1,1,H,W,D).cuda())>0.9).float()
    dist,idx = edt((mask[0,0,::2,::2,::2]==0).squeeze().cpu().numpy(),return_indices=True)
    moving_r = F.interpolate((moving[::2,::2,::2].cuda().reshape(-1)[idx[0]*104*96+idx[1]*104+idx[2]]).unsqueeze(0).unsqueeze(0),scale_factor=2,mode='trilinear')
    moving_r.view(-1)[mask.view(-1)!=0] = moving.cuda().reshape(-1)[mask.view(-1)!=0]
    #moving_r = moving.cuda().view(1,1,H,W,D)*mask


    #compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        mindssc_fix = losses.MINDSSC(fixed_r,1,2).half()#*fixed_mask.cuda().half()#.cpu()
        mindssc_mov = losses.MINDSSC(moving_r,1,2).half()#*moving_mask.cuda().half()#.cpu()

        mind_fix = F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)
        mind_mov = F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp)


    ssd,ssd_argmin = correlate(mind_fix,mind_mov,disp_hw,grid_sp)
    disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

    disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp)

    #disp_soft = F.avg_pool3d(disp_mesh_t.view(3,-1)[:,ssd_argmin.view(-1)].reshape(1,3,H//grid_sp,W//grid_sp,D//grid_sp),3,padding=1,stride=1)

    #ssd_,ssd_argmin_ = correlate(mind_mov,mind_fix,disp_hw,grid_sp)
    #disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp)
    scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2
    #disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=10)


    disp_hr = F.interpolate(disp_soft*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)



    grid_sp = 2

    with torch.no_grad():
        mind_fix = F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)
        mind_mov = F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp)


        #patch_mind_fix = nn.Flatten(5,)(F.pad(mind_fix,(1,1,1,1,1,1)).unfold(2,3,1).unfold(3,3,1).unfold(4,3,1)).permute(0,1,5,2,3,4).reshape(1,-1,H//grid_sp,W//grid_sp,D//grid_sp)
        #patch_mind_mov = nn.Flatten(5,)(F.pad(mind_mov,(1,1,1,1,1,1)).unfold(2,3,1).unfold(3,3,1).unfold(4,3,1)).permute(0,1,5,2,3,4).reshape(1,-1,H//grid_sp,W//grid_sp,D//grid_sp)


    #create optimisable displacement grid
    grid_sp = 2
    disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)


    net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
    net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    #torch.cuda.synchronize()
    #t0 = time.time()
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)

    #run Adam optimisation with diffusion regularisation and B-spline smoothing
    lambda_weight = .65# with tps: .5, without:0.7
    for iter in range(50):#80
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
        reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
        lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
        lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

        #grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/torch.tensor([63/2,63/2,68/2]).unsqueeze(0).cuda()).flip(1)

        scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
        grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

        patch_mov_sampled = F.grid_sample(mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=False,mode='bilinear')#,padding_mode='border')
        #patch_mov_sampled_sq = F.grid_sample(mind_mov.pow(2).float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=True,mode='bilinear')
        #sampled_cost = (patch_mov_sampled_sq-2*patch_mov_sampled*mind_fix+mind_fix.pow(2)).mean(1)*12

        sampled_cost = (patch_mov_sampled-mind_fix).pow(2).mean(1)*12
        #sampled_cost = F.grid_sample(ssd2.view(-1,1,17,17,17).float(),disp_sample.view(-1,1,1,1,3)/disp_hw,align_corners=True,padding_mode='border')
        loss = sampled_cost.mean()
        (loss+reg_loss).backward()
        optimizer.step()

    fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
    disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1),3,padding=1,stride=1),3,padding=1,stride=1)

    disp_field = F.interpolate(disp_smooth,scale_factor = 0.5,mode='trilinear',align_corners=False)


    torch.cuda.synchronize()
    t0b = time.time()
    x1 = disp_field[0,0,:,:,:].cpu().float().data.numpy()
    y1 = disp_field[0,1,:,:,:].cpu().float().data.numpy()
    z1 = disp_field[0,2,:,:,:].cpu().float().data.numpy()

    #x1 = zoom(x,1/2,order=2).astype('float16')
    #y1 = zoom(y,1/2,order=2).astype('float16')
    #z1 = zoom(z,1/2,order=2).astype('float16')


    np.savez_compressed('/data_supergrover2/heinrich/L2R2021/convexAdam/submission/task_02/disp_'+str(nu).zfill(4)+'_'+str(nu).zfill(4)+'.npz',np.stack((x1,y1,z1),0))

    torch.cuda.synchronize()
    #t1 = time.time()
    #print(t1-t0,'sec (optim)')
    t1 = time.time()
    print(t1-t0,'time all sec',t0a-t0+t1-t0b,'read/write')
    time_all[nu-21] = t1-t0

print('time all',time_all.mean())
torch.save(time_all,'/data_supergrover2/heinrich/L2R2021/convexAdam/task2_times.pth')
