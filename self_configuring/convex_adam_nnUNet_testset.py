import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from convexAdam.convex_adam_utils import (correlate, coupled_convex,
                                          inverse_consistency)

warnings.filterwarnings("ignore")


# extract MIND and/or semantic nnUNet features
# extract MIND and/or semantic nnUNet features
def extract_features(pred_fixed,
                    pred_moving):

    eps=1e-32
    H,W,D = pred_fixed.shape[-3:]
    
    #weight = 1/((torch.bincount(pred_fixed.long().reshape(-1))+torch.bincount(pred_moving.long().reshape(-1)))+eps).float().pow(.3)
    #weight /= weight.mean()
    
    combined_bins = torch.bincount(pred_fixed.long().reshape(-1))+torch.bincount(pred_moving.long().reshape(-1))
    
    pos = torch.nonzero(combined_bins).reshape(-1)
    
    #features_fix = 10*(pred_fixed[:,1:,:,:,:].data.float().permute(0,1,4,3,2).contiguous()*weight[1:].view(1,-1,1,1,1).cuda()).half()
    #features_mov = 10*(pred_moving[:,1:,:,:,:].data.float().permute(0,1,4,3,2).contiguous()*weight[1:].view(1,-1,1,1,1).cuda()).half()
    
    pred_fixed = F.one_hot(pred_fixed.cuda().view(1,H,W,D).long())[:,:,:,:,pos]
    pred_moving = F.one_hot(pred_moving.cuda().view(1,H,W,D).long())[:,:,:,:,pos]
    
    weight = 1/((torch.bincount(pred_fixed.permute(0,4,1,2,3).argmax(1).long().reshape(-1))+torch.bincount(pred_moving.permute(0,4,1,2,3).argmax(1).long().reshape(-1)))+eps).float().pow(.3)
    weight /= weight.mean()

    features_fix = 10*(pred_fixed.data.float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
    features_mov = 10*(pred_moving.data.float().permute(0,4,1,2,3).contiguous()*weight.view(1,-1,1,1,1).cuda()).half()
    return features_fix, features_mov

# coupled convex optimisation with adam instance optimisation
def convex_adam(img_fixed,
                img_moving,
                pred_fixed,
                pred_moving,
                use_mask,
                mask_fixed,
                mask_moving,
                lambda_weight,
                grid_sp,
                disp_hw,
                selected_niter,
                selected_smooth,
                grid_sp_adam=2,
                ic=True):
    
    H,W,D = img_fixed.shape

    torch.cuda.synchronize()
    t0 = time.time()

    #compute features and downsample (using average pooling)
    with torch.no_grad():      
        
        features_fix, features_mov = extract_features(pred_fixed=pred_fixed,
                                                        pred_moving=pred_moving)
        
        features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)
        
        n_ch = features_fix_smooth.shape[1]

    # compute correlation volume with SSD
    ssd,ssd_argmin = correlate(features_fix_smooth,features_mov_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

    # provide auxiliary mesh grid
    disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)
    
    # perform coupled convex optimisation
    disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))
    
    # if "ic" flag is set: make inverse consistent
    if ic:
        scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2

        ssd_,ssd_argmin_ = correlate(features_mov_smooth,features_fix_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

        disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
        disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)

        disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
    
    else:
        disp_hr=disp_soft

    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)


        #create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)


        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)

        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)

        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

            scale = torch.tensor([(H//grid_sp_adam-1)/2,(W//grid_sp_adam-1)/2,(D//grid_sp_adam-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),grid_disp.view(1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam,3).cuda(),align_corners=False,mode='bilinear')

            sampled_cost = (patch_mov_sampled-patch_features_fix).pow(2).mean(1)*12
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()


        fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
        disp_hr = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth//2
            disp_hr = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)


        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth//2
            disp_hr = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1),kernel_smooth,padding=padding_smooth,stride=1)

        
    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1-t0
    print('case time: ', case_time)

    x = disp_hr[0,0,:,:,:].cpu().half().data.numpy()
    y = disp_hr[0,1,:,:,:].cpu().half().data.numpy()
    z = disp_hr[0,2,:,:,:].cpu().half().data.numpy()
    displacements = np.stack((x,y,z),3).astype(float)

    return displacements, case_time
