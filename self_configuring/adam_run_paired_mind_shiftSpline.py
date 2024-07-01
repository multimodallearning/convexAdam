import time
import warnings
import nibabel as nib
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import sys
warnings.filterwarnings("ignore")
import os

import cupy
from cupyx.scipy.ndimage import distance_transform_edt
from tqdm.auto import trange,tqdm

from convexAdam_hyper_util import MINDSSC, correlate, coupled_convex, inverse_consistency, dice_coeff,extract_features, sort_rank, jacobian_determinant_3d, kovesi_spline, GaussianSmoothing, gpu_usage, extract_features_nnunet,cupy_hd95

                        
def get_data_train(topk,HWD,f_img,f_key,f_mask):
    l2r_base_folder = './'
    H,W,D = HWD[0],HWD[1],HWD[2]
    
    imgs_fixed = []
    keypts_fixed = []
    masks_fixed = []
    imgs_moving = []
    keypts_moving = []
    masks_moving = []

    for i in tqdm(topk):
        file_img = f_img.replace('xxxx',str(i).zfill(4))
        file_key = f_key.replace('xxxx',str(i).zfill(4))
        file_mask = f_mask.replace('xxxx',str(i).zfill(4))
        
        img_fixed = torch.from_numpy(nib.load(file_img).get_fdata()).float().cuda().contiguous()
        key_fixed = torch.from_numpy(np.loadtxt(file_key,delimiter=',')).float()
        mask_fixed = torch.from_numpy(nib.load(file_mask).get_fdata()).float().cuda().contiguous()
        imgs_fixed.append(img_fixed)
        keypts_fixed.append(key_fixed)
        masks_fixed.append(mask_fixed)

        file_img = file_img.replace('0000',str(1).zfill(4))
        file_key = file_key.replace('0000',str(1).zfill(4))
        file_mask = file_mask.replace('0000',str(1).zfill(4))
        
        img_moving = torch.from_numpy(nib.load(file_img).get_fdata()).float().cuda().contiguous()
        key_moving = torch.from_numpy(np.loadtxt(file_key,delimiter=',')).float()
        mask_moving = torch.from_numpy(nib.load(file_mask).get_fdata()).float().cuda().contiguous()
        imgs_moving.append(img_moving)
        keypts_moving.append(key_moving)
        masks_moving.append(mask_moving)

    return imgs_fixed,keypts_fixed,masks_fixed,imgs_moving,keypts_moving,masks_moving

def main(gpunum,configfile,convex_s):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    topk = config['topk']
    
    print('using 15 registration pairs')
    imgs_fixed,keypts_fixed,masks_fixed,imgs_moving,keypts_moving,masks_moving = get_data_train(topk,config['HWD'],config['f_img'],config['f_key'],config['f_mask'])
    robust30 = []
    for i in range(len(topk)):
        tre0 = (keypts_fixed[i]-keypts_moving[i]).square().sum(-1).sqrt()
        robust30.append(tre0.topk(int(len(tre0)*.3),largest=True).indices)

    torch.manual_seed(1004)
    settings = (torch.rand(100,4)*torch.tensor([3,3,4,6])+torch.tensor([0.5,0.5,1.5,1.5])).round()
    #print(settings[1])
    settings[settings[:,2]==2,3] = torch.minimum(settings[settings[:,2]==2,3],torch.tensor([5]))

    print(settings.min(0).values,settings.max(0).values,)

    mind_r = int(settings[convex_s,0])#1
    mind_d = int(settings[convex_s,1])#1
    grid_sp = int(settings[convex_s,2])#6
    disp_hw = int(settings[convex_s,3])#4

    
    print('using predetermined setting s=',convex_s)
    
    print('setting mind_r',mind_r,'mind_d',mind_d,'grid_sp',grid_sp,'disp_hw',disp_hw)
    tre_convex = torch.zeros(len(topk))
    ##APPLY BEST CONVEX TO TRAIN
    disps_lr = []
    for i in trange(len(topk)):

        t0 = time.time()

        img_fixed = imgs_fixed[i].cuda()
        key_fixed = keypts_fixed[i].cuda()
        mask_fixed = masks_fixed[i].cuda()
        
        img_moving = imgs_moving[i].cuda()
        key_moving = keypts_moving[i].cuda()
        mask_moving = masks_moving[i].cuda()

        H, W, D = img_fixed.shape[-3:]
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
        torch.cuda.synchronize()
        t0 = time.time()

        # compute features and downsample (using average pooling)
        with torch.no_grad():

            features_fix,features_mov = extract_features(img_fixed,img_moving,mind_r,mind_d,True,mask_fixed,mask_moving)

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
            
            disp_hr = F.interpolate(disp_lr,size=(H,W,D),mode='trilinear',align_corners=False)
            t2 = time.time()
            disps_lr.append(disp_lr.cpu())
            scale1 = torch.tensor([D-1,W-1,H-1]).cuda()/2

        lms_fix1 = (key_fixed.flip(1)/scale1-1).cuda().view(1,-1,1,1,3)
        disp_sampled = F.grid_sample(disp_hr.float().cuda(),lms_fix1).squeeze().t().cpu().data
        #TRE0 = (key_fixed.cpu()-key_moving.cpu()).square().sum(-1).sqrt()
        TRE1 = (key_fixed.cpu()-key_moving.cpu()+disp_sampled).square().sum(-1).sqrt()
        tre_convex[i] = TRE1.mean()
        #print(TRE0.mean(),'>',TRE1.mean())
    print('TRE convex',tre_convex.mean())
    del disp_soft; del disp_soft_; del ssd_; del ssd; del disp_hr; del features_fix; del features_mov; del features_fix_smooth; del features_mov_smooth;

    ##FIND OPTIMAL ADAM SETTING

    avgs = [GaussianSmoothing(.7).cuda(),\
        GaussianSmoothing(1).cuda(),kovesi_spline(1.3,4).cuda(),kovesi_spline(1.6,4).cuda(),kovesi_spline(1.9,4).cuda(),kovesi_spline(2.2,4).cuda(),kovesi_spline(2.5,4).cuda(),kovesi_spline(2.8,4).cuda()]


    torch.manual_seed(2004)
    #settings_adam = (torch.rand(50,5)*torch.tensor([2,2,3,5,7])+torch.tensor([0.5,0.5,0.5,.5,1.5])).round() #new

#    settings_adam = (torch.rand(50,5)*torch.tensor([2,2,4,5,7])+torch.tensor([0.5,0.5,0.5,-.49,1.5])).round()
    settings_adam = (torch.rand(75,5)*torch.tensor([2,2,4,5,7])+torch.tensor([0.5,0.5,0.5,.5,1.5])).round()
    settings_adam[:,4] *= .2
    #settings_adam[0] = torch.tensor([1,2,2,3,1.5])
    #settings_adam[1] = torch.tensor([1,2,1,4,1.5])
    #print('s0',settings_adam[0])
    #print('s1',settings_adam[1])
    #settings[settings[:,2]==2,3] = torch.minimum(settings[settings[:,2]==2,3],torch.tensor([5]))
    #print(settings[1])
    torch.cuda.empty_cache()
    print(settings_adam.min(0).values,settings_adam.max(0).values,gpu_usage())

    jstd2 = torch.zeros(75,4,4,2)
    tre2 = torch.zeros(75,4,4,2)
    tre_min = 100
    for s in trange(75):
        for i in trange(len(topk)):

            t0 = time.time()

            img_fixed = imgs_fixed[i].cuda()
            key_fixed = keypts_fixed[i].cuda()
            mask_fixed = masks_fixed[i].cuda()
            
            img_moving = imgs_moving[i].cuda()
            key_moving = keypts_moving[i].cuda()
            mask_moving = masks_moving[i].cuda()
            
            mind_r = int(settings_adam[s,0])#1
            mind_d = int(settings_adam[s,1])#2
            grid_sp_adam = int(settings_adam[s,2])#6
            avg_n = int(settings_adam[s,3])#6
            if(grid_sp_adam==1):
                avg_n += 2
            if(grid_sp_adam==2):
                avg_n += 1
            lambda_weight = float(settings_adam[s,4])#4

            t0 = time.time()
            

            H, W, D = img_fixed.shape[-3:]
            grid0_hr = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
            torch.cuda.synchronize()
            t0 = time.time()

            # compute features and downsample (using average pooling)
            with torch.no_grad():
                features_fix,features_mov = extract_features(img_fixed,img_moving,mind_r,mind_d,True,mask_fixed,mask_moving)
                n_ch = features_mov.shape[1]
            # run Adam instance optimisation
            with torch.no_grad():
                patch_features_fix = F.avg_pool3d(features_fix,grid_sp_adam,stride=grid_sp_adam)
                patch_features_mov = F.avg_pool3d(features_mov,grid_sp_adam,stride=grid_sp_adam)

            disp_hr = F.interpolate(disps_lr[i].cuda().float(),size=(H,W,D),mode='trilinear',align_corners=False)
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
                                lms_fix1 = (key_fixed.flip(1)/scale1-1).cuda().view(1,-1,1,1,3)
                                disp_sampled = F.grid_sample(disp_hr.float().cuda(),lms_fix1).squeeze().t().cpu().data
                                jac_det = jacobian_determinant_3d(disp_hr.float(),False)

                                TRE1 = (key_fixed.cpu()-key_moving.cpu()+disp_sampled).square().sum(-1).sqrt()

                                #t_mind[s] += t1-t0
                                #t_convex[s] += t2-t1
                                tre2[s,ii,kk,0] += 1/len(topk)*TRE1.mean()
                                tre2[s,ii,kk,1] += 1/len(topk)*TRE1[robust30[i]].mean()
                                jac_det_log = jac_det.add(3).clamp_(0.000000001, 1000000000).log()#.std()
                                jstd2[s,ii,kk,0] += 1/len(topk)*(jac_det_log).std().cpu()
                                jstd2[s,ii,kk,1] += 1/len(topk)*((jac_det<0).float().mean()).cpu()
                                

        torch.save([tre2,jstd2],config['output_adam'])
        loss.cpu(); del loss
        reg_loss.cpu(); del reg_loss;
        net.cpu(); del net;
        if(tre2[s,:,:,0].min()<tre_min):
            print('s',s,'%0.3f'%(tre2[s,:,:,0].min().item()),'%0.3f'%(tre2[s,:,:,1].min().item()),'jstd','%0.3f'%(jstd2[s,:,:,0].mean().item()))
            tre_min = tre2[s,:,:,0].min()
    rank2 = sort_rank(tre2[:,...,0].reshape(-1))
    rank2 *= sort_rank(tre2[:,...,1].reshape(-1))
    rank2 *= sort_rank(jstd2[:,...,0].reshape(-1))

    rank2 = rank2.pow(1/3)
    print(rank2.argmax()//16,rank2.argmax()//len(settings_adam))
    print(tre2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()//len(settings_adam)],jstd2[:].reshape(len(settings_adam),16,2)[rank2.argmax()//16,rank2.argmax()//len(settings_adam)])

    print(settings_adam[int(rank2.argmax())//16])
    
    torch.save([rank2,tre2,jstd2],config['output_adam'])
    #tensor(3) tensor(1)
#tensor([0.6857, 0.5596]) tensor([0.1263, 0.0077])
#tensor([3.0000, 2.0000, 1.6000])

        

    use_mask = True
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    configfile = (sys.argv[2])
    convex_s = int(sys.argv[3])
    main(gpu_id,configfile,convex_s)
