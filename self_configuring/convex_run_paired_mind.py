import sys
import time
import warnings

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import json
import os

from convexAdam_hyper_util import (correlate, coupled_convex, extract_features,
                                   inverse_consistency,
                                   jacobian_determinant_3d, sort_rank)
from tqdm.auto import tqdm, trange


def get_data_train(topk,HWD,f_img,f_key,f_mask):
    l2r_base_folder = './'
#~/storage/staff/christophgrossbroeh/data/Learn2Reg/Learn2Reg_Dataset_release_v1.1
    #topk = (1,2,3,4,5,16,17,18,19,20)
    

# ####   topk = (3,5,6,10,11,12,13,16)
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

def main(gpunum,configfile):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    topk = config['topk']


    #starting full run  0  out of 100
    #   setting mind_r 3 mind_d 2 grid_sp 3 disp_hw 7
    #100%|███████████████████████████████████████████████████████████████████████████████████| 15/15 [00:38<00:00,  2.59s/it]
    #s 0 1.623 1.865 jstd tensor(0.0968)
    #tensor(2)
    #tensor([1.9585, 2.1685]) tensor([0.0729, 0.0000]) tensor(3.5369)
    #tensor([2., 2., 4., 4.])

    
    #topk_pair = config['topk_pair']# ((2,4),(4,9),(3,4),(0,4),(1,4),(4,7),(4,5),(2,8))
    #topk_pair = []
    #for i in range(0,10):
    #    for j in range(0,10):
    #        if(i<j):
    #            topk_pair.append((i,j))
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
    
    t_mind = torch.zeros(100)
    t_convex = torch.zeros(100)
    tre = torch.zeros(100,2)
    jstd = torch.zeros(100,2)
    tre_min = 1000
    for s in range(100):
        mind_r = int(settings[s,0])#1
        mind_d = int(settings[s,1])#1
        grid_sp = int(settings[s,2])#6
        disp_hw = int(settings[s,3])#4

        print('starting full run ',s,' out of 100')
        print('setting mind_r',mind_r,'mind_d',mind_d,'grid_sp',grid_sp,'disp_hw',disp_hw)
        for i in trange(len(topk)):

            t0 = time.time()

            img_fixed = imgs_fixed[i].cuda()
            key_fixed = keypts_fixed[i].cuda()
            mask_fixed = masks_fixed[i].cuda()
            
            img_moving = imgs_moving[i].cuda()
            key_moving = keypts_moving[i].cuda()
            mask_moving = masks_moving[i].cuda()

            H, W, D = img_moving.shape
            grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
            torch.cuda.synchronize()
            t0 = time.time()

            # compute features and downsample (using average pooling)
            with torch.no_grad():

                # compute features and downsample (using average pooling)
                
                features_fix,features_mov = extract_features(img_fixed,img_moving,mind_r,mind_d,True,mask_fixed,mask_moving)

                features_fix_smooth = F.avg_pool3d(features_fix,grid_sp,stride=grid_sp)
                features_mov_smooth = F.avg_pool3d(features_mov,grid_sp,stride=grid_sp)

                n_ch = features_fix_smooth.shape[1]
                t1 = time.time()
                #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # compute correlation volume with SSD
                torch.cuda.empty_cache()
                ssd,ssd_argmin = correlate(features_fix_smooth,features_mov_smooth,disp_hw,grid_sp,(H,W,D), n_ch)

                # provide auxiliary mesh grid
                disp_mesh_t = F.affine_grid(disp_hw*torch.eye(3,4).cuda().half().unsqueeze(0),(1,1,disp_hw*2+1,disp_hw*2+1,disp_hw*2+1),align_corners=True).permute(0,4,1,2,3).reshape(3,-1,1)

                # perform coupled convex optimisation
                disp_soft = coupled_convex(ssd,ssd_argmin,disp_mesh_t,grid_sp,(H,W,D))

                # if "ic" flag is set: make inverse consistent
                scale = torch.tensor([H//grid_sp-1,W//grid_sp-1,D//grid_sp-1]).view(1,3,1,1,1).cuda().half()/2
                torch.cuda.empty_cache()

                ssd_,ssd_argmin_ = correlate(features_mov_smooth,features_fix_smooth,disp_hw,grid_sp,(H,W,D), n_ch)
                
                disp_soft_ = coupled_convex(ssd_,ssd_argmin_,disp_mesh_t,grid_sp,(H,W,D))
                disp_ice,_ = inverse_consistency((disp_soft/scale).flip(1),(disp_soft_/scale).flip(1),iter=15)

                disp_hr = F.interpolate(disp_ice.flip(1)*scale*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)
                t2 = time.time()

                scale1 = torch.tensor([D-1,W-1,H-1]).cuda()/2
                lms_fix1 = (key_fixed.flip(1)/scale1-1).cuda().view(1,-1,1,1,3)
                disp_sampled = F.grid_sample(disp_hr.float().cuda(),lms_fix1).squeeze().t().cpu().data
                jac_det = jacobian_determinant_3d(disp_hr.float(),False)
                torch.cuda.empty_cache()
                
            TRE1 = (key_fixed.cpu()-key_moving.cpu()+disp_sampled).square().sum(-1).sqrt()
            
            t_mind[s] += t1-t0
            t_convex[s] += t2-t1
            tre[s,0] += 1/len(topk)*TRE1.mean()
            tre[s,1] += 1/len(topk)*TRE1[robust30[i]].mean()
            jac_det_log = jac_det.add(3).clamp_(0.000000001, 1000000000).log()#.std()
            jstd[s,0] += 1/len(topk)*(jac_det_log).std().cpu()
            jstd[s,1] += 1/len(topk)*((jac_det<0).float().mean()).cpu()
            

        if(tre[s,0]<tre_min):
            print('s',s,'%0.3f'%tre[s,0].item(),'%0.3f'%tre[s,1].item(),'jstd',jstd[s,0])
            tre_min = tre[s,0]
            
    rank1 = sort_rank(tre[:,0])
    rank1 *= sort_rank(tre[:,1])

    rank1 *= sort_rank(jstd[:,0])#.sqrt()

    rank1 = rank1.pow(1/3)
    print(rank1.argmax())
    print(tre[rank1.argmax()],jstd[rank1.argmax()],t_convex[rank1.argmax()])
    print(settings[rank1.argmax()])
    torch.save([rank1,tre,jstd,t_convex],config['output'])

        

    use_mask = True
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    configfile = str(sys.argv[2])
    main(gpu_id,configfile)
