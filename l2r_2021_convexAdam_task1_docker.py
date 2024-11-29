
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
import sys
import time

from scipy.ndimage import distance_transform_edt as edt

A = torch.ones(64,64).cuda()
A.requires_grad = True
A.sum().backward()
sys.path.append('voxelmorph/pytorch/')
import losses

print(losses.mind_loss)

def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated()*1e-9, torch.cuda.max_memory_allocated()*1e-9))

    
import pandas as pd


def util_kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape
   
    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1)/torch.tensor([W, H, D], device=device)
    
    return kpts_pt_
def convert_crop_field(case,fix_affine_disp_p):
    df = pd.read_csv('/data_supergrover1/hansen/temp/reg/data/abdomen/mr_ct/cases.csv')
    df = df.loc[df['Id'] == case]
    ref_spacing = torch.ones(3)*2#'2x2x2'
    flip = 'xy'
    fix_shape = torch.from_numpy(np.array([float(j) for j in df['FixShape'].values[0][1:-1].split(' ')])).float()
    fix_spacing = torch.from_numpy(np.array([float(j) for j in df['FixSpacing'].values[0][1:-1].split(' ')])).float()
    fix_crop = torch.from_numpy(np.array([float(j) for j in df['FixCrop'].values[0][1:-1].split(' ')])).float().view(3, 2).permute(1, 0)
    
    mov_shape = torch.from_numpy(np.array([float(j) for j in df['MovShape'].values[0][1:-1].split(' ')])).float()
    mov_spacing = torch.from_numpy(np.array([float(j) for j in df['MovSpacing'].values[0][1:-1].split(' ')])).float()
    mov_crop = torch.from_numpy(np.array([float(j) for j in df['MovCrop'].values[0][1:-1].split(' ')])).float().view(3, 2).permute(1, 0)
    
    #fix_affine_disp_p = torch.from_numpy(nib.load(disp_path).get_fdata()).unsqueeze(0).float()
    
    fix_scale_factor = (fix_spacing / ref_spacing)
    new_shape = ((fix_crop[1, :] - fix_crop[0, :]) * fix_scale_factor).round()
    new_fix_scale_factor = new_shape / (fix_crop[1, :] - fix_crop[0, :])
    new_fix_spacing = fix_spacing / new_fix_scale_factor
    new_mov_scale_factor = new_shape / (mov_crop[1, :] - mov_crop[0, :])
    new_mov_spacing = mov_spacing / new_mov_scale_factor

    fix_affine = torch.tensor([[1/new_fix_scale_factor[0], 0, 0, fix_crop[0, 0]],
                               [0, 1/new_fix_scale_factor[1], 0, fix_crop[0, 1]],
                               [0, 0, 1/new_fix_scale_factor[2], fix_crop[0, 2]],
                               [0, 0, 0, 1]])

    mov_affine = torch.tensor([[1/new_mov_scale_factor[0], 0, 0, mov_crop[0, 0]],
                               [0, 1/new_mov_scale_factor[1], 0, mov_crop[0, 1]],
                               [0, 0, 1/new_mov_scale_factor[2], mov_crop[0, 2]],
                               [0, 0, 0, 1]])

    fix_grid = torch.stack(torch.meshgrid(torch.arange(fix_shape[0]).cuda(),
                                          torch.arange(fix_shape[1]).cuda(),
                                          torch.arange(fix_shape[2]).cuda(), indexing='ij'), dim=3).view(1,-1,3)

    fix_grid_affine = torch.matmul(fix_affine.inverse().cuda(), torch.cat([fix_grid[0], torch.ones(fix_grid.shape[1], 1).cuda()], dim=1).t()).t()[:,:3].unsqueeze(0)

    fix_grid_affine_pt =util_kpts_pt(fix_grid_affine, new_shape, align_corners=True)
    fix_grid_affine_disp_p = F.grid_sample(fix_affine_disp_p.permute(0, 4, 1, 2, 3), fix_grid_affine_pt.view(1, 1, 1, -1, 3), mode='bilinear', padding_mode='border', align_corners=True).permute(0, 4, 2, 3, 1).view(1, -1, 3)

    fix_grid_affine_p = fix_grid_affine * new_fix_spacing.cuda()

    mov_grid_affine_est_p = fix_grid_affine_p + fix_grid_affine_disp_p

    mov_grid_affine_est = mov_grid_affine_est_p / new_mov_spacing.cuda()

    mov_grid_est = torch.matmul(mov_affine.cuda(), torch.cat([mov_grid_affine_est[0].cuda(), torch.ones(mov_grid_affine_est.shape[1], 1).cuda()], dim=1).t()).t()[:,:3].unsqueeze(0)

    
    disp = mov_grid_est - fix_grid
    disp = disp.view(1, *(fix_shape.long().tolist()), 3)

    if 'x' in flip:
        disp = disp.flip(1)
        disp[:, :, :, :, 0] = - disp[:, :, :, :, 0]
        
    if 'y' in flip:
        disp = disp.flip(2)
        disp[:, :, :, :, 1] = - disp[:, :, :, :, 1]
        
    if 'z' in flip:
        disp = disp.flip(3)
        disp[:, :, :, :, 2] = - disp[:, :, :, :, 2]
    
    disp = F.interpolate(disp.permute(0, 4, 1, 2, 3),scale_factor=0.5,mode='trilinear',align_corners=False)[0].cpu().numpy().astype(np.float16)
#    disp = np.array([zoom(disp.permute(0, 4, 1, 2, 3)[0][i].numpy(), 0.5, order=2) for i in range(3)]).astype(np.float16)
    return disp




#correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(mind_fix,mind_mov,disp_hw,grid_sp,shape):
    H = int(shape[0]); W = int(shape[1]); D = int(shape[2]);

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
            ssd[i::(disp_hw*2+1)] = F.avg_pool3d(F.avg_pool3d(mind_sum.transpose(2,1),3,stride=1,padding=1),3,stride=1,padding=1).squeeze(1)
        ssd = ssd.view(disp_hw*2+1,disp_hw*2+1,disp_hw*2+1,H//grid_sp,W//grid_sp,D//grid_sp).transpose(1,0).reshape((disp_hw*2+1)**3,H//grid_sp,W//grid_sp,D//grid_sp)
        ssd_argmin = torch.argmin(ssd,0)#
        #ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    t1 = time.time()
    print(t1-t0,'sec (ssd)')
    gpu_usage()
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





#folder1 = '/share/data_supergrover1/heinrich/L2R2021/Task1/'
folder = '/share/data_supergrover1/hansen/temp/reg/data/abdomen/mr_ct/preprocessed/crop+resize/'

device = 'cuda'


#TCIA12_img_fixed.nii.gz    TCIA12_seg_moving.nii.gz   TCIA14_seg_fixed.nii.gz    TCIA16_mask_moving.nii.gz
time_all = torch.zeros(3)
dice_all = torch.zeros(3)
jac_all = torch.zeros(3)

case_time = torch.zeros(8)

torch.cuda.synchronize()
t0_ = time.time()
for nu in range(1,16,2):#14,16
    folder = '/share/data_supergrover1/hansen/temp/reg/data/abdomen/mr_ct/preprocessed/crop+resize/'
    img_fixed = torch.from_numpy(nib.load(folder+'TCIA'+str(nu).zfill(2)+'_img_fixed.nii.gz').get_fdata()).float()
    H,W,D = img_fixed.shape
    img_moving = torch.from_numpy(nib.load(folder+'TCIA'+str(nu).zfill(2)+'_img_moving.nii.gz').get_fdata()).float()
    fixed_mask = (torch.from_numpy(nib.load(folder+'TCIA'+str(nu).zfill(2)+'_mask_fixed.nii.gz').get_fdata())>.5).float()
    grid_sp = 4#4
    disp_hw = 8#6

    print(img_fixed.shape,img_moving.shape,fixed_mask.shape)
    torch.cuda.synchronize()
    t0 = time.time()

    #compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        mindssc_fix = losses.MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).cuda(),1,2).half()#*fixed_mask.cuda().half()#.cpu()
        mindssc_mov = losses.MINDSSC(img_moving.unsqueeze(0).unsqueeze(0).cuda(),1,2).half()#*moving_mask.cuda().half()#.cpu()

        mind_fix = F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)
        mind_mov = F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp)


    ssd,ssd_argmin = correlate(mind_fix,mind_mov,disp_hw,grid_sp,(H,W,D))
    disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)
    disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))
    scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2
    ssd_,ssd_argmin_ = correlate(mind_mov,mind_fix,disp_hw,grid_sp,(H,W,D))
    disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
    disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)

    disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)


    grid_sp = 3

    with torch.no_grad():

        patch_mind_fix = F.avg_pool3d(mindssc_fix,grid_sp,stride=grid_sp)
        patch_mind_mov = F.avg_pool3d(mindssc_mov,grid_sp,stride=grid_sp)


    #create optimisable displacement grid
    disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)


    net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
    net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    #torch.cuda.synchronize()
    #t0 = time.time()
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)

    #run Adam optimisation with diffusion regularisation and B-spline smoothing
    lambda_weight = .6# with tps: .5, without:0.7
    for iter in range(40):#80
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
        reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
        lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
        lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

        #grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/torch.tensor([63/2,63/2,68/2]).unsqueeze(0).cuda()).flip(1)

        scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).cuda().unsqueeze(0)
        grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

        patch_mov_sampled = F.grid_sample(patch_mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=False,mode='bilinear')#,padding_mode='border')
        #patch_mov_sampled_sq = F.grid_sample(patch_mind_mov.pow(2).float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3).cuda(),align_corners=True,mode='bilinear')
        #sampled_cost = (patch_mov_sampled_sq-2*patch_mov_sampled*patch_mind_fix+patch_mind_fix.pow(2)).mean(1)*12

        sampled_cost = (patch_mov_sampled-patch_mind_fix).pow(2).mean(1)*12
        #sampled_cost = F.grid_sample(ssd2.view(-1,1,17,17,17).float(),disp_sample.view(-1,1,1,1,3)/disp_hw,align_corners=True,padding_mode='border')
        loss = sampled_cost.mean()
        (loss+reg_loss).backward()
        optimizer.step()

    fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
    #fitted_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(fitted_grid,3,padding=1,stride=1),3,padding=1,stride=1),3,padding=1,stride=1)
    disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
    
    if(True):
        ident = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//3,W//3,D//3),align_corners=True)
        disp = disp_hr.cpu().float().permute(0,2,3,4,1)/torch.Tensor([H-1,W-1,D-1]).view(1,1,1,1,3)*2
        disp = disp.flip(4)
        #print(ident.shape)
        fixed_mask3 = fixed_mask[1::3,1::3,1::3][:ident.shape[1],:ident.shape[2],:ident.shape[3]]
        ident1 = ident.view(-1,3)[fixed_mask3.reshape(-1)>0,:]
        #print(ident1.shape)
        ident_mask = ident1[torch.randperm(int((fixed_mask3>0).sum()))[:2048*2]]
        disp_sampled = F.grid_sample(disp.cuda().permute(0,4,1,2,3),ident_mask.view(1,-1,1,1,3)).squeeze(3).squeeze(3).permute(0,2,1)

        torch.cuda.synchronize()
        t0_ = time.time()
        dense_flow_ = thin_plate_dense(ident_mask.unsqueeze(0), disp_sampled, (H, W, D), 4, 0)
        dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).cuda().view(1,3,1,1,1)/2
        torch.cuda.synchronize()
        t1_ = time.time()
        print(t1_-t0_,'sec (TPS)')
   
    
    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(dense_flow,3,padding=1,stride=1),3,padding=1,stride=1),3,padding=1,stride=1)
    
    disp = disp_smooth.cuda().float().permute(0,2,3,4,1)/torch.tensor([H-1,W-1,D-1]).cuda().view(1,1,1,1,3)*2
    disp = disp.flip(4)
    
    fix_spacing = torch.tensor(nib.load(folder+'TCIA'+str(nu).zfill(2)+'_img_fixed.nii.gz').header.get_zooms(), device=device)
    mov_spacing = torch.tensor(nib.load(folder+'TCIA'+str(nu).zfill(2)+'_img_moving.nii.gz').header.get_zooms(), device=device)
    img_fix = img_fixed.unsqueeze(0).unsqueeze(0)
    img_fix_grid = torch.stack(torch.meshgrid(torch.arange(img_fix.shape[2], device=device),torch.arange(img_fix.shape[3], device=device),torch.arange(img_fix.shape[4], device=device), indexing='ij'), dim=3).unsqueeze(0)
    img_mov_warped_grid = img_fix_grid + disp_smooth.permute(0,2,3,4,1)
    img_fix_grid_p = img_fix_grid * fix_spacing
    img_mov_warped_grid_p = img_mov_warped_grid * mov_spacing
    disp_p = img_mov_warped_grid_p - img_fix_grid_p
    disp_npz = convert_crop_field('TCIA'+str(nu).zfill(2),disp_p.cuda())
    
    np.savez_compressed('/data_supergrover2/heinrich/L2R2021/convexAdam/submission/task_01/disp_'+str(nu).zfill(4)+'_'+str(nu).zfill(4)+'.npz',disp_npz)


    torch.cuda.synchronize()
    t1 = time.time()
    print('time all',t1-t0)
    #time_all[nu//2-6] = t1-t0

    case_time[(nu-1)//2] = t1-t0

torch.cuda.synchronize()
t1_ = time.time()
print('total time',t1_-t0_)
torch.save(case_time,'/data_supergrover2/heinrich/L2R2021/convexAdam/task1_times.pth')
