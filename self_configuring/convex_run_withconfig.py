import sys
import time
import warnings

import nibabel as nib
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
import json
import os

from convexAdam_hyper_util import (correlate, coupled_convex, cupy_hd95,
                                   dice_coeff, extract_features_nnunet,
                                   inverse_consistency,
                                   jacobian_determinant_3d, sort_rank)
from tqdm.auto import trange


def get_data_train(topk,HWD,f_predict,f_gt):
    l2r_base_folder = './'
#~/storage/staff/christophgrossbroeh/data/Learn2Reg/Learn2Reg_Dataset_release_v1.1
    #topk = (1,2,3,4,5,16,17,18,19,20)
    

# ####   topk = (3,5,6,10,11,12,13,16)
    H,W,D = HWD[0],HWD[1],HWD[2]
    #robustify
    preds_fixed = []
    segs_fixed = []
    
    for i in topk:
        pred_fixed = torch.from_numpy(nib.load(f_predict.replace('xxxx',str(i).zfill(4))).get_fdata()).float().cuda().contiguous()
        seg_fixed = torch.from_numpy(nib.load(f_gt.replace('xxxx',str(i).zfill(4))).get_fdata()).float().cuda().contiguous()
        segs_fixed.append(seg_fixed)
        #img_fixed =  torch.from_numpy(nib.load(l2r_base_folder+'AbdomenCTCT/imagesTr/AbdomenCTCT_00'+str(i).zfill(2)+'_0000.nii.gz').get_fdata()).float().cuda().contiguous()
        preds_fixed.append(pred_fixed)
    return preds_fixed,segs_fixed

def main(gpunum,configfile):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str((gpunum))
    print(torch.cuda.get_device_name())

    with open(configfile, 'r') as f:
        config = json.load(f)
    topk = config['topk']

    num_labels = config['num_labels']-1
    topk_pair = config['topk_pair']# ((2,4),(4,9),(3,4),(0,4),(1,4),(4,7),(4,5),(2,8))
    #topk_pair = []
    #for i in range(0,10):
    #    for j in range(0,10):
    #        if(i<j):
    #            topk_pair.append((i,j))
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
    print(settings.min(0).values,settings.max(0).values,)
    
    t_mind = torch.zeros(100)
    t_convex = torch.zeros(100)
    dice = torch.zeros(100,2)
    jstd = torch.zeros(100,2)
    hd95 = torch.zeros(100)
    dice_min = 0
    for s in range(100):
        nn_mult = int(settings[s,0])#1
        grid_sp = int(settings[s,1])#6
        disp_hw = int(settings[s,2])#4

        print('starting full run ',s,' out of 100')
        print('setting nn_mult',nn_mult,'grid_sp',grid_sp,'disp_hw',disp_hw)
        for i in trange(len(topk_pair)):
            ij = topk_pair[i]

            t0 = time.time()

            pred_fixed = preds_fixed[ij[0]].float()
            pred_moving = preds_fixed[ij[1]].float()
            seg_fixed = segs_fixed[ij[0]]
            seg_moving = segs_fixed[ij[1]]

            H, W, D = pred_fixed.shape
            grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)
            torch.cuda.synchronize()
            t0 = time.time()

            # compute features and downsample (using average pooling)
            with torch.no_grad():

                features_fix, features_mov = extract_features_nnunet(pred_fixed=pred_fixed,
                                                              pred_moving=pred_moving,mult=nn_mult)

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

                #lms_fix1 = (lms_fixed.flip(1)/scale1-1).cuda().view(1,-1,1,1,3)
                #disp_sampled = F.grid_sample(disp_hr.float().cuda(),lms_fix1).squeeze().t().cpu().data
                jac_det = jacobian_determinant_3d(disp_hr.float(),False)
                torch.cuda.empty_cache()
                
            seg_warped = F.grid_sample(seg_moving.view(1,1,H,W,D),grid0+disp_hr.permute(0,2,3,4,1).flip(-1).div(scale1),mode='nearest').squeeze()
            DICE1 = dice_coeff(seg_fixed,seg_warped,num_labels+1)
            HD95 = cupy_hd95(seg_fixed.long(),seg_warped.long(),num_labels)

            #print(TRE0.mean(),'>',TRE1.mean())
            t_mind[s] += t1-t0
            t_convex[s] += t2-t1
            dice[s,0] += 1/len(topk_pair)*DICE1.mean()
            dice[s,1] += 1/len(topk_pair)*DICE1[robust30[i]].mean()
            jac_det_log = jac_det.add(3).clamp_(0.000000001, 1000000000).log()#.std()
            jstd[s,0] += 1/len(topk_pair)*(jac_det_log).std().cpu()
            jstd[s,1] += 1/len(topk_pair)*((jac_det<0).float().mean()).cpu()
            hd95[s] += 1/len(topk_pair)*(HD95).mean().cpu()
            

        torch.save([dice,jstd,hd95,t_convex],config['output'])

        if(dice[s,0]>dice_min):
            print('s',s,'%0.3f'%dice[s,0].item(),'%0.3f'%dice[s,1].item(),'jstd',jstd[s,0])
            dice_min = dice[s,0]
            
    rank1 = sort_rank(-dice[:,0])
    rank1 *= sort_rank(-dice[:,1])

    rank1 *= sort_rank(hd95[:])
    rank1 *= sort_rank(jstd[:,0])#.sqrt()

    rank1 = rank1.pow(1/4)
    print(rank1.argmax())
    print(dice[rank1.argmax()],jstd[rank1.argmax()],t_convex[rank1.argmax()])
    print(settings[rank1.argmax()])
    torch.save([rank1,dice,jstd,hd95,t_convex],config['output'])

        

    use_mask = True
if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    configfile = str(sys.argv[2])
    main(gpu_id,configfile)
