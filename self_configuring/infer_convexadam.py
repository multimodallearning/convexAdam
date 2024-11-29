import json
import sys
import time
import warnings

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import os

from convexAdam_hyper_util import (GaussianSmoothing, correlate,
                                   coupled_convex, extract_features_nnunet,
                                   inverse_consistency, kovesi_spline)
from tqdm.auto import trange


def get_data_train(topk,HWD,f_predict,f_gt):
    l2r_base_folder = './'
    print('reading test data')
    f_predict = f_predict.replace('sTr/','sTs/')
    f_gt = f_gt.replace('sTr/','sTs/')
    
    H,W,D = HWD[0],HWD[1],HWD[2]
    #robustify
    preds_fixed = []
    segs_fixed = []
    preds_moving = []
    segs_moving = []

    for i in topk:
        file_pred = f_predict.replace('xxxx',str(i).zfill(4))
        #file_seg = f_gt.replace('xxxx',str(i).zfill(4)) #not available for test-scans
        
        pred_fixed = torch.from_numpy(nib.load(file_pred).get_fdata()).float().cuda().contiguous()
        #seg_fixed = torch.from_numpy(nib.load(file_seg).get_fdata()).float().cuda().contiguous()
        segs_fixed.append(None)#seg_fixed)
        preds_fixed.append(pred_fixed)
        
        
                

    return preds_fixed,segs_fixed
def main(gpunum,configfile,convex_s,adam_s1,adam_s2):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    topk = config['test']

    num_labels = config['num_labels']-1
    eval_labels = num_labels
    topk_pair = config['test_pair']
    
    print('using all test',len(topk_pair),' registration pairs')
    preds_fixed,segs_fixed = get_data_train(topk,config['HWD'],config['f_predict'],config['f_gt'])


    torch.manual_seed(1004)
    settings = (torch.rand(100,3)*torch.tensor([6,4,6])+torch.tensor([.5,1.5,1.5])).round()
    #print(settings[1])
    settings[:,0] *= 2.5
    settings[settings[:,1]==2,2] = torch.minimum(settings[settings[:,1]==2,2],torch.tensor([5]))
    print(settings.min(0).values,settings.max(0).values,)
    
    print('using predetermined setting s=',convex_s)
    
    nn_mult = int(settings[convex_s,0])#1
    grid_sp = int(settings[convex_s,1])#6
    disp_hw = int(settings[convex_s,2])#4
    print('setting nn_mult',nn_mult,'grid_sp',grid_sp,'disp_hw',disp_hw)

    ##APPLY BEST CONVEX TO TRAIN
    disps_lr = []
    for i in trange(len(topk_pair)):

        t0 = time.time()

        pred_fixed = preds_fixed[int(topk_pair[i][0])].float()
        pred_moving = preds_fixed[int(topk_pair[i][1])].float()
        
        H, W, D = pred_fixed.shape[-3:]
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
        torch.cuda.synchronize()
        t0 = time.time()

        # compute features and downsample (using average pooling)
        with torch.no_grad():

            features_fix, features_mov = extract_features_nnunet(pred_fixed=pred_fixed,
                                                          pred_moving=pred_moving)

            features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
            features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)

            n_ch = features_fix_smooth.shape[1]
            t1 = time.time()
            #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # compute correlation volume with SSD
            ssd,ssd_argmin = correlate(features_fix_smooth,features_mov_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

            # provide auxiliary mesh grid
            disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

            # perform coupled convex optimisation
            disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))

            # if "ic" flag is set: make inverse consistent
            scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2

            ssd_,ssd_argmin_ = correlate(features_mov_smooth,features_fix_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

            disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
            disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)
            disp_lr = disp_ice.flip(1)*scale*grid_sp
            disps_lr.append(disp_lr.data.cpu())
            disp_hr = F.interpolate(disp_lr,size=(H,W,D),mode='trilinear',align_corners=False)
            t2 = time.time()
            scale1 = torch.tensor([D-1,W-1,H-1]).cuda()/2

        
    del disp_soft; del disp_soft_; del ssd_; del ssd; del disp_hr; del features_fix; del features_mov; del features_fix_smooth; del features_mov_smooth;

        
    ##FIND OPTIMAL ADAM SETTING
    avgs = [GaussianSmoothing(.7).cuda(),GaussianSmoothing(1).cuda(),kovesi_spline(1.3,4).cuda(),kovesi_spline(1.6,4).cuda(),kovesi_spline(1.9,4).cuda(),kovesi_spline(2.2,4).cuda()]

    torch.manual_seed(2004)
    settings_adam = (torch.rand(75,3)*torch.tensor([4,5,7])+torch.tensor([0.5,.5,1.5])).round()
    settings_adam[:,2] *= .2


    torch.cuda.empty_cache()


    grid_sp_adam = int(settings_adam[adam_s1,0])#6
    avg_n = int(settings_adam[adam_s1,1])#6
        ##SHIFT-SPLINE
    if(grid_sp_adam==1):
        avg_n += 2
    if(grid_sp_adam==2):
        avg_n += 1

    lambda_weight = float(settings_adam[adam_s1,2])#4

    iters = (adam_s2//4)*20+60
    kks = (adam_s2%4)
    print('setting grid_sp_adam',grid_sp_adam,'avg_n',avg_n,'lambda_weight',lambda_weight,'iters',iters,'kks',kks)

    jstd2 = torch.zeros(len(topk_pair),2)
    dice2 = torch.zeros(len(topk_pair),eval_labels)
    hd95_2 = torch.zeros(len(topk_pair),eval_labels)

    dice_ident = torch.zeros(len(topk_pair),eval_labels)

    for i in trange(len(topk_pair)):
        file = config['f_gt'].split('/')[-1]
        stem = file.split('_')[0]
        file_field = stem+'/fieldsTs/'+file.replace(stem,'disp').replace('0000',str(int(topk[topk_pair[i][1]])).zfill(4)).replace('xxxx',str(int(topk[topk_pair[i][0]])).zfill(4))
        print('writing output-nii to ',file_field)
        t0 = time.time()

        
        pred_fixed = preds_fixed[int(topk_pair[i][0])].float()
        pred_moving = preds_fixed[int(topk_pair[i][1])].float()
        #seg_fixed = segs_fixed[int(topk_pair[i][0])].float()
        #seg_moving = segs_fixed[int(topk_pair[i][1])].float()
        #mind_r = int(settings_adam[s,0])#1
        #mind_d = int(settings_adam[s,1])#2
        
        t0 = time.time()
        

        H, W, D = pred_fixed.shape[-3:]
        grid0_hr = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
        torch.cuda.synchronize()
        t0 = time.time()

        # compute features and downsample (using average pooling)
        with torch.no_grad():

            features_fix, features_mov = extract_features_nnunet(pred_fixed=pred_fixed,
                                                        pred_moving=pred_moving)


            n_ch = features_fix.shape[1]
        # run Adam instance optimisation
        with torch.no_grad():
            patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)

        disp_hr = F.interpolate(disps_lr[i].float().cuda(),size=(H,W,D),mode='trilinear',align_corners=False)

        #create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)

        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1)

        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)
        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(iters):
            optimizer.zero_grad()

            disp_sample = (avgs[avg_n](net[0].weight)).permute(0,2,3,4,1)#,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
            reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
            lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
            lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()

            scale = torch.tensor([(H//grid_sp_adam-1)/2,(W//grid_sp_adam-1)/2,(D//grid_sp_adam-1)/2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1,3).cuda().float()+((disp_sample.view(-1,3))/scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),grid_disp.view(1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam,3).cuda(),align_corners=False,mode='bilinear')

            sampled_cost = (patch_mov_sampled-patch_features_fix).pow(2).mean(1)*n_ch
            loss = sampled_cost.mean()
            (loss+reg_loss).backward()
            optimizer.step()
            scale1 = torch.tensor([D-1,W-1,H-1]).cuda()/2

        with torch.no_grad():
    
            fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
            disp_hr = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)

            kernel_smooth = 3; padding_smooth = kernel_smooth//2
            
            for kk in range(kks):
                if(kk>0):
                    disp_hr = F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1)
            #save displacement field
            nib.save(nib.Nifti1Image(disp_hr.permute(0,2,3,4,1).squeeze().data.cpu().numpy(),np.eye(4)),file_field)
            
        

    use_mask = True
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    configfile = (sys.argv[2])
    convex_s = int(sys.argv[3])
    adam_s1 = int(sys.argv[4])
    adam_s2 = int(sys.argv[5])
    main(gpu_id,configfile,convex_s,adam_s1,adam_s2)
