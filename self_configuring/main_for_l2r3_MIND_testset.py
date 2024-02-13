import argparse
import json
import os
import warnings

import nibabel as nib
import torch
from convex_adam_MIND_testset import *

warnings.filterwarnings("ignore")


def main(task_name, 
        mind_r,
        mind_d,
        use_mask,
        lambda_weight,
        grid_sp,
        disp_hw,
        selected_niter,
        selected_smooth,
        data_dir,
        result_path):

    task_dir = os.path.join(data_dir,task_name)
    dataset_json = os.path.join(task_dir,task_name+'_dataset.json')
    
    with open(dataset_json, 'r') as f:
             data = json.load(f)
    val_pairs = data['registration_test']
    
    # create save directory
    save_paths = ['results_testset']
    for save_path in save_paths:
        new_path = os.path.join(result_path, task_name, save_path)
        isExist = os.path.exists(new_path)
        if not isExist:
            os.makedirs(new_path)
        files = os.listdir(new_path)
        for item in files:
            if item.endswith(".nii.gz"):
                os.remove(os.path.join(new_path, item))
            if item.endswith(".nii"):
                os.remove(os.path.join(new_path, item))

    
    case_times = torch.zeros(len(val_pairs))
    ii=0
    for _, pair in enumerate(val_pairs):
        path_fixed = os.path.join(task_dir, pair['fixed'])
        path_moving = os.path.join(task_dir, pair['moving'])
        img_fixed = torch.from_numpy(nib.load(path_fixed).get_fdata()).float()
        img_moving = torch.from_numpy(nib.load(path_moving).get_fdata()).float()
        if use_mask:
            path_fixed_mask = os.path.join(task_dir, pair['fixed'].replace('images','masks'))
            path_moving_mask = os.path.join(task_dir, pair['moving'].replace('images','masks'))
            mask_fixed = torch.from_numpy(nib.load(path_fixed_mask).get_fdata()).float()
            mask_moving = torch.from_numpy(nib.load(path_moving_mask).get_fdata()).float()
        else: 
            mask_fixed = None
            mask_moving = None


        displacements, case_time = convex_adam(img_fixed=img_fixed, 
                                            img_moving=img_moving,
                                            mind_r=mind_r,
                                            mind_d=mind_d,
                                            use_mask=use_mask,
                                            mask_fixed=mask_fixed,
                                            mask_moving=mask_moving,
                                            lambda_weight=lambda_weight, 
                                            grid_sp=grid_sp, 
                                            disp_hw=disp_hw,
                                            selected_niter=selected_niter,
                                            selected_smooth=selected_smooth)

        case_times[ii] = case_time
        ii+=1
        
        affine = nib.load(path_fixed).affine

        disp_path = os.path.join(result_path, task_name, 'results_testset', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_nii = nib.Nifti1Image(displacements, affine)
        nib.save(disp_nii, disp_path)

    median_case_time = case_times.median().item()
    print('median case time: ', median_case_time)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--lambda_weight', type=float, required=True)
    parser.add_argument('--grid_sp', type=int, required=True)
    parser.add_argument('--disp_hw', type=int, required=True)
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--selected_niter', type=int, required=True)
    parser.add_argument('--selected_smooth', type=int, required=True)
    parser.add_argument('--use_mask', choices=('True','False'), default= 'False')
    parser.add_argument('--data_dir', type=str, default='/share/data_zoe3/grossbroehmer/Learn2Reg2022/Learn2Reg_Dataset_v11/')
    parser.add_argument('--result_path', type=str, default='/share/data_abby2/hsiebert/code/adam_optimisation/JournalExperiments/l2r2022/results/')
    args = parser.parse_args()

    if args.use_mask == 'True':
        use_mask=True
    else:
        use_mask=False

    task_name = args.task_name       
    data_dir = args.data_dir
    mind_r = args.mind_r
    mind_d = args.mind_d
    lambda_weight = args.lambda_weight
    grid_sp = args.grid_sp
    disp_hw = args.disp_hw
    result_path = args.result_path
    selected_niter = args.selected_niter
    selected_smooth = args.selected_smooth

    main(task_name,
        mind_r,
        mind_d,
        use_mask,
        lambda_weight,
        grid_sp,
        disp_hw,
        selected_niter,
        selected_smooth,
        data_dir,
        result_path)