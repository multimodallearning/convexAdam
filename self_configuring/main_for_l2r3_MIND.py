import argparse
import json
import os
import warnings
from pathlib import Path

import nibabel as nib
import torch
from convex_adam_MIND import *
from L2R_main.evaluation import evaluation

warnings.filterwarnings("ignore")


def main(task_name, 
        mind_r,
        mind_d,
        use_mask,
        lambda_weight,
        grid_sp,
        disp_hw,
        evaluate,
        data_dir,
        result_path,
        config_path):

    task_dir = os.path.join(data_dir,task_name)
    dataset_json = os.path.join(task_dir,task_name+'_dataset.json')
    
    with open(dataset_json, 'r') as f:
             data = json.load(f)
    val_pairs = data['registration_val']

    if len(data['modality'].keys()) == 1:
        modality_fixed = data['modality']['0']
        modality_moving = data['modality']['0']
    
    if len(data['modality'].keys()) == 2:
        modality_fixed = data['modality']['0']
        modality_moving = data['modality']['1']

    if len(data['modality'].keys()) == 3:
        modality_fixed = data['modality']['0']
        modality_moving = data['modality']['2']
    
    outstr = '_'+'MIND'+str(int(mind_r))+str(int(mind_d))+'_'+str(int(lambda_weight*100))+'lambda_'+str(grid_sp)+'gs1_'+str(disp_hw)+'disp_'+str(use_mask)+'Masks'
    print(outstr)

    print('>>> Modality fixed: ', modality_fixed)
    print('>>> Modality moving: ', modality_moving)
    print('>>> Settings: lambda_weight: {}; grid_sp: {}; disp_hw: {}'.format(lambda_weight, grid_sp, disp_hw))
    
    # create save directory
    save_paths = ['40_smoothing0', '60_smoothing0', '80_smoothing0', '40_smoothing3', '60_smoothing3', '80_smoothing3', '40_smoothing5', '60_smoothing5', '80_smoothing5']
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


        displacements_40, displacements_60, displacements_80, displacements_40_smooth3, displacements_60_smooth3, displacements_80_smooth3, displacements_40_smooth5, displacements_60_smooth5, displacements_80_smooth5, case_time = convex_adam(img_fixed=img_fixed, 
                                            img_moving=img_moving,
                                            mind_r=mind_r,
                                            mind_d=mind_d,
                                            use_mask=use_mask,
                                            mask_fixed=mask_fixed,
                                            mask_moving=mask_moving,
                                            lambda_weight=lambda_weight, 
                                            grid_sp=grid_sp, 
                                            disp_hw=disp_hw)

        case_times[ii] = case_time
        ii+=1
        
        
        affine = nib.load(path_fixed).affine

        # save smoothing 0
        #disp_path = os.path.join(result_path, task_name, '40_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '40_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_40, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '60_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '60_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_60, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '80_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '80_smoothing0', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_80, affine)
        nib.save(disp_nii, disp_path)

        # save smoothing 3
        #disp_path = os.path.join(result_path, task_name, '40_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '40_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_40_smooth3, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '60_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '60_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_60_smooth3, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '80_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '80_smoothing3', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_80_smooth3, affine)
        nib.save(disp_nii, disp_path)

        # save smoothing 5
        #disp_path = os.path.join(result_path, task_name, '40_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '40_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_40_smooth5, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '60_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '60_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_60_smooth5, affine)
        nib.save(disp_nii, disp_path)

        #disp_path = os.path.join(result_path, task_name, '80_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii.gz'))
        disp_path = os.path.join(result_path, task_name, '80_smoothing5', 'disp_{}_{}'.format(pair['fixed'][-16:-12], pair['moving'][-16:-12]+'.nii'))
        disp_nii = nib.Nifti1Image(displacements_80_smooth5, affine)
        nib.save(disp_nii, disp_path)


    median_case_time = case_times.median().item()
    print('median case time: ', median_case_time)
    
    if evaluate:
        print('>>>EVALUATION')
        DEFAULT_GROUND_TRUTH_PATH = Path(task_dir)
        eval_config_path = config_path+task_name+'_VAL_evaluation_config.json'

        for save_path in save_paths:
            DEFAULT_INPUT_PATH = Path(os.path.join(result_path, task_name,save_path))
            DEFAULT_EVALUATION_OUTPUT_FILE_PATH = Path(os.path.join(result_path, task_name,save_path)+'/metrics'+outstr+'.json')
            evaluation.evaluate_L2R(DEFAULT_INPUT_PATH, DEFAULT_GROUND_TRUTH_PATH, DEFAULT_EVALUATION_OUTPUT_FILE_PATH, eval_config_path, verbose=False)
            
            with open(DEFAULT_EVALUATION_OUTPUT_FILE_PATH, "r") as jsonFile:
                data = json.load(jsonFile)
            
            data[task_name]["aggregates"]["median_case_time"] = median_case_time

            with open(DEFAULT_EVALUATION_OUTPUT_FILE_PATH, "w") as jsonFile:
                json.dump(data, jsonFile)

            print('Path to evaluation JSON file: ', DEFAULT_EVALUATION_OUTPUT_FILE_PATH)
    
    else:
        print('NO EVALUATION')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--lambda_weight', type=float, required=True)
    parser.add_argument('--grid_sp', type=int, required=True)
    parser.add_argument('--disp_hw', type=int, required=True)
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--use_mask', choices=('True','False'), default= 'False')
    parser.add_argument('--evaluate', choices=('True','False'), default= 'True')
    parser.add_argument('--data_dir', type=str, default='/share/data_zoe3/grossbroehmer/Learn2Reg2022/Learn2Reg_Dataset_v11/')
    parser.add_argument('--result_path', type=str, default='/share/data_abby2/hsiebert/code/adam_optimisation/JournalExperiments/l2r2022/results/')
    parser.add_argument('--config_path', type=str, default='/share/data_abby2/hsiebert/code/adam_optimisation/JournalExperiments/l2r2022/L2R_main/evaluation/evaluation_configs/')
    args = parser.parse_args()

        
    if args.evaluate == 'True':
        evaluate=True
    else:
        evaluate=False

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
    config_path = args.config_path

    main(task_name,
        mind_r,
        mind_d,
        use_mask,
        lambda_weight,
        grid_sp,
        disp_hw,
        evaluate,
        data_dir,
        result_path,
        config_path)