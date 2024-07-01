import time
import warnings
import nibabel as nib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import sys
warnings.filterwarnings("ignore")
import os
import json
import cupy
from cupyx.scipy.ndimage import distance_transform_edt
from tqdm.auto import trange,tqdm

from convexAdam_hyper_util import MINDSSC, correlate, coupled_convex, inverse_consistency, dice_coeff,extract_features, sort_rank, jacobian_determinant_3d, kovesi_spline, GaussianSmoothing, gpu_usage, extract_features_nnunet,cupy_hd95

            
def get_data_train(topk,HWD,f_predict,f_gt):
    l2r_base_folder = './'
#~/storage/staff/christophgrossbroeh/data/Learn2Reg/Learn2Reg_Dataset_release_v1.1
    #topk = (1,2,3,4,5,16,17,18,19,20)
    

# ####   topk = (3,5,6,10,11,12,13,16)
    H,W,D = HWD[0],HWD[1],HWD[2]
    #robustify
    preds_fixed = []
    segs_fixed = []
    
    for i in tqdm(topk):
        pred_fixed = torch.from_numpy(nib.load(f_predict.replace('xxxx',str(i).zfill(4))).get_fdata()).float().cuda().contiguous()
        seg_fixed = torch.from_numpy(nib.load(f_gt.replace('xxxx',str(i).zfill(4))).get_fdata()).float().cuda().contiguous()
        segs_fixed.append(seg_fixed)
        #img_fixed =  torch.from_numpy(nib.load(l2r_base_folder+'AbdomenCTCT/imagesTr/AbdomenCTCT_00'+str(i).zfill(2)+'_0000.nii.gz').get_fdata()).float().cuda().contiguous()
        preds_fixed.append(pred_fixed)
    return preds_fixed,segs_fixed

def main(gpunum,configfile,convex_s):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    topk = config['topk']

    num_labels = config['num_labels']-1
    topk_pair = config['topk_pair']
    
    print('using ',len(topk_pair),'registration pairs')
    preds_fixed,segs_fixed = get_data_train(topk,config['HWD'],config['f_predict'],config['f_gt'])
    robust30 = []
    for ij in topk_pair:
        dice = dice_coeff(segs_fixed[ij[0]],segs_fixed[ij[1]],num_labels+1)
        robust30.append(dice.topk(max(1,int(config['num_labels']*.3)),largest=False).indices)


    torch.manual_seed(1004)
    settings = (torch.rand(100,3)*torch.tensor([6,4,6])+torch.tensor([.5,1.5,1.5])).round()
    #print(settings[1])
    settings[:,0] *= 2.5
    settings[settings[:,1]==2,2] = torch.minimum(settings[settings[:,1]==2,2],torch.tensor([5]))
    #print(settings.min(0).values,settings.max(0).values,)
    
    print('using predetermined setting s=',convex_s)
    
    nn_mult = int(settings[convex_s,0])#1
    grid_sp = int(settings[convex_s,1])#6
    disp_hw = int(settings[convex_s,2])#4
    print('setting nn_mult',nn_mult,'grid_sp',grid_sp,'disp_hw',disp_hw)

    ##APPLY BEST CONVEX TO TRAIN
#    disps_hr = []
    disps_lr = []
    for i in range(len(topk_pair)):
        ij = topk_pair[i]

        t0 = time.time()

        pred_fixed = preds_fixed[ij[0]].float()
        pred_moving = preds_fixed[ij[1]].float()
        seg_fixed = segs_fixed[ij[0]]
        seg_moving = segs_fixed[ij[1]]


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
            disp_lr = (disp_ice.flip(1)*scale*grid_sp)
            disps_lr.append(disp_lr.data.cpu().half())
            disp_hr = F.interpolate(disp_lr,size=(H,W,D),mode='trilinear',align_corners=False)
            t2 = time.time()
            #disps_hr.append(disp_hr)
            scale1 = torch.tensor([D-1,W-1,H-1]).cuda()/2

        DICE0 = dice_coeff(seg_fixed,seg_moving,num_labels+1)
        seg_warped = F.grid_sample(seg_moving.view(1,1,H,W,D),grid0+disp_hr.permute(0,2,3,4,1).flip(-1).div(scale1),mode='nearest').squeeze()
        DICE1 = dice_coeff(seg_fixed,seg_warped,num_labels+1)
        print(DICE0.mean(),'>',DICE1.mean())
    del disp_soft; del disp_soft_; del ssd_; del ssd; del disp_hr; del features_fix; del features_mov; del features_fix_smooth; del features_mov_smooth;
        
    ##FIND OPTIMAL ADAM SETTING

    avgs = [GaussianSmoothing(.7).cuda(),\
        GaussianSmoothing(1).cuda(),kovesi_spline(1.3,4).cuda(),kovesi_spline(1.6,4).cuda(),kovesi_spline(1.9,4).cuda(),kovesi_spline(2.2,4).cuda(),kovesi_spline(2.5,4).cuda(),kovesi_spline(2.8,4).cuda()]


    torch.manual_seed(2004)
    settings_adam = (torch.rand(75,3)*torch.tensor([4,5,7])+torch.tensor([0.5,.5,1.5])).round()

    #settings_adam = (torch.rand(50,3)*torch.tensor([4,4,7])+torch.tensor([1.5,-.49,1.5])).round()
    settings_adam[:,2] *= .2
    #print(settings_adam[1])
    #settings[settings[:,2]==2,3] = torch.minimum(settings[settings[:,2]==2,3],torch.tensor([5]))
    #print(settings[1])
    torch.cuda.empty_cache()
    print(settings_adam.min(0).values,settings_adam.max(0).values,gpu_usage())
    all_jac = torch.zeros(75,len(topk_pair),4,4)
    jstd2 = torch.zeros(75,4,4,2)
    dice2 = torch.zeros(75,4,4,2)
    hd95_2 = torch.zeros(75,4,4)
    dice2_min = 0
    for s in trange(75):
        for i in range(len(topk_pair)):
            ij = topk_pair[i]
            
            #mind_r = int(settings_adam[s,0])#1
            #mind_d = int(settings_adam[s,1])#2
            grid_sp_adam = int(settings_adam[s,0])#6
            avg_n = int(settings_adam[s,1])#6
            ##SHIFT-SPLINE
            if(grid_sp_adam==1):
                avg_n += 2
            if(grid_sp_adam==2):
                avg_n += 1
            
            lambda_weight = float(settings_adam[s,2])#4

            t0 = time.time()
            
            pred_fixed = preds_fixed[ij[0]].float()
            pred_moving = preds_fixed[ij[1]].float()
            seg_fixed = segs_fixed[ij[0]]
            seg_moving = segs_fixed[ij[1]]


            H, W, D = pred_fixed.shape[-3:]
            grid0_hr = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
            torch.cuda.synchronize()
            t0 = time.time()

            # compute features and downsample (using average pooling)
            with torch.no_grad():      

                features_fix, features_mov = extract_features_nnunet(pred_fixed=pred_fixed,
                                                            pred_moving=pred_moving)


                n_ch = features_mov.shape[1]
            # run Adam instance optimisation
            with torch.no_grad():
                patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
                patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)

            disp_lr = disps_lr[i]
            disp_hr = F.interpolate(disp_lr.float().cuda(),size=(H,W,D),mode='trilinear',align_corners=False)

            #create optimisable displacement grid
            disp_lr = F.interpolate(disp_hr,size=(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),mode='trilinear',align_corners=False)

            net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),bias=False))
            net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
            net.cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=1)

            grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H//grid_sp_adam,W//grid_sp_adam,D//grid_sp_adam),align_corners=False)
            #run Adam optimisation with diffusion regularisation and B-spline smoothing
            for iter in range(120):
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
                #lms_fix1 = (lms_fixed.flip(1)/scale1-1).cuda().view(1,-1,1,1,3)

                if(iter>=59):
                    with torch.no_grad():
                        if((iter-59)%20==0):

                            fitted_grid = disp_sample.detach().permute(0,4,1,2,3)
                            disp_hr = F.interpolate(fitted_grid*grid_sp_adam,size=(H,W,D),mode='trilinear',align_corners=False)

                            kernel_smooth = 3; padding_smooth = kernel_smooth//2
                            
                            ii = int((iter-59)//20)
                            for kk in range(4):
                                if(kk>0):
                                    disp_hr = F.avg_pool3d(disp_hr,kernel_smooth,padding=padding_smooth,stride=1)
                                #disp_sampled = F.grid_sample(disp_hr.float().cuda(),lms_fix1).squeeze().t().cpu().data   
                                #TRE1 = ((lms_fixed+disp_sampled-lms_moving)*torch.tensor([1.75, 1.25, 1.75])).square().sum(-1).sqrt()

                                seg_warped = F.grid_sample(seg_moving.view(1,1,H,W,D),grid0_hr+disp_hr.permute(0,2,3,4,1).flip(-1).div(scale1),mode='nearest').squeeze()
                                DICE1 = dice_coeff(seg_fixed,seg_warped,num_labels+1)
        
                                dice2[s,ii,kk,0] += 1/len(topk_pair)*DICE1.mean()
                                dice2[s,ii,kk,1] += 1/len(topk_pair)*DICE1[robust30[i]].mean()
                                jac_det = jacobian_determinant_3d(disp_hr.float(),False).reshape(-1)
                                jac_det_log = jac_det.add(3).clamp_(0.000000001, 1000000000).log()#.std()
                                all_jac[s,i,ii,kk] = (jac_det_log).std().cpu()
                                jstd2[s,ii,kk,0] += 1/len(topk_pair)*(jac_det_log).std().cpu()
                                jstd2[s,ii,kk,1] += 1/len(topk_pair)*((jac_det<0).float().mean()).cpu()
                                
                                HD95 = cupy_hd95(seg_fixed.long(),seg_warped.long(),num_labels)

                                hd95_2[s,ii,kk] += 1/len(topk_pair)*(HD95).mean().cpu()

        torch.save([dice2,jstd2,hd95_2],config['output_adam'])
        torch.save(all_jac,config['output_jac'])

        if(dice2[s].max()>dice2_min):
            print('s',s,'%0.3f'%dice2[s,...,0].max().item(),'%0.3f'%dice2[s,...,1].max().item(),'jstd',jstd2[s,...,0].mean())
            dice2_min = dice2[s].max()
    rank2 = sort_rank(-dice2[:,...,0].reshape(-1))
    rank2 *= sort_rank(-dice2[:,...,1].reshape(-1))
    rank2 *= sort_rank(jstd2[:,...,0].reshape(-1))
    rank2 *= sort_rank(hd95_2[:].reshape(-1))

    rank2 = rank2.pow(1/4)
    print(rank2.argmax()//16,'old',rank2.argmax()//len(settings_adam),'corrected',rank2.argmax()%16)
    print('old',dice2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()//len(settings_adam)],jstd2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()//len(settings_adam)])
    print('corrected',dice2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()%16],jstd2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()%16])

    #tensor(36) tensor(11)
    #tensor([0.7171, 0.5609]) tensor([0.1063, 0.0032])
    #print(settings_adam[int(rank2.argmax())//16])
    print(settings_adam[int(rank2.argmax())//16])
    #rank1 = sort_rank(-dice[:,0])
    #rank1 *= sort_rank(-dice[:,1])

    #rank1 *= sort_rank(hd95[:])
    #rank1 *= sort_rank(jstd[:,0])#.sqrt()

    #rank1 = rank1.pow(1/4)
    #print(rank1.argmax())
    #print(dice[rank1.argmax()],jstd[rank1.argmax()],t_convex[rank1.argmax()])
    #print(settings[rank1.argmax()])
    torch.save([rank2,dice2,jstd2,hd95_2],config['output_adam'])
    #tensor(3) tensor(1)
#tensor([0.6857, 0.5596]) tensor([0.1263, 0.0077])
#tensor([3.0000, 2.0000, 1.6000])

        

    use_mask = True
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    configfile = str(sys.argv[2])
    convex_s = int(sys.argv[3])
    main(gpu_id,configfile,convex_s)
