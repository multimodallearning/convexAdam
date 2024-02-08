import argparse
import json
import os

import main_for_l2r3_MIND
import main_for_l2r3_MIND_testset
import main_for_l2r3_nnUNet
import main_for_l2r3_nnUNet_testset
import numpy as np
import scipy.stats
import torch


def sc_convex_adam(args):
    experiments = {}

    main_save_path = os.path.join(args.result_path, args.task_name)
    
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    save_paths = ['40_smoothing0', '60_smoothing0', '80_smoothing0', '40_smoothing3', '60_smoothing3', '80_smoothing3', '40_smoothing5', '60_smoothing5', '80_smoothing5']

    # check if folder with masks provided
    mask_path = os.path.join(args.data_dir, args.task_name, 'masksTr')
    if os.path.exists(mask_path):
        use_mask = True
        print('masks provided')
    else:
        use_mask = False
        print('no masks provided')

    config_path =os.path.join(args.data_dir, args.task_name)+'/'

    eval_config_path = config_path+args.task_name+'_VAL_evaluation_config.json'
    with open(eval_config_path, 'r') as f:
        eval_config_data = json.load(f)
        
    n_metrics = len(eval_config_data['evaluation_methods'])
    smooth_metric = eval_config_data['evaluation_methods'][0]['name']
    
    sim_metric_1 = eval_config_data['evaluation_methods'][1]['name']
    if n_metrics > 2:
        sim_metric_2 = eval_config_data['evaluation_methods'][2]['name']
    else:
        sim_metric_2 = None

    experiments['metrics'] = {}
    experiments['metrics']['smooth'] = smooth_metric
    experiments['metrics']['sim1'] = sim_metric_1
    experiments['metrics']['sim2'] = sim_metric_2

    expected_shape = eval_config_data['expected_shape']
    expected_vol = expected_shape[0]*expected_shape[1]*expected_shape[2]

    vol_limit = 1000000
    if expected_vol>vol_limit:
        large_volume = True
        print('apply settings for large volumes')
    else:
        large_volume = False
        print('apply settings for small volumes')

    # hyperparameter options that should be examined
    if large_volume:
        options_grid_sp = [6]
        options_disp_hw = [6,4]
    else:
        options_grid_sp = [4]
        options_disp_hw = [4,2]
    options_lambda_weight = [0.75,1.0,1.25]
    


    task_dir = os.path.join(args.data_dir,args.task_name)
    dataset_json = os.path.join(task_dir,args.task_name+'_dataset.json')

    
    with open(dataset_json, 'r') as f:
             data = json.load(f)

    if data['provided_data']['0'][1] == 'label':
        semantic_features = True
        print('semantic features available')
    else: 
        semantic_features = False
        print('no semantic features available')

    if len(data['modality'].keys()) == 1:
        modality_fixed = data['modality']['0']
        modality_moving = data['modality']['0']
    
    if len(data['modality'].keys()) == 2:
        modality_fixed = data['modality']['0']
        modality_moving = data['modality']['1']

    if ('US' in modality_fixed) or ('US' in modality_moving):
        selected_mind_r = 3
        selected_mind_d = 3
    else:
        selected_mind_r = 1
        selected_mind_d = 2


    # ABLATIONS    
    for grid_sp in options_grid_sp:
        for disp_hw in options_disp_hw:
            for w_lambda in options_lambda_weight:
                main_for_l2r3_MIND.main(task_name=args.task_name, 
                                    mind_r=selected_mind_r,
                                    mind_d=selected_mind_d,
                                    use_mask=use_mask,
                                    lambda_weight=w_lambda,
                                    grid_sp=grid_sp,
                                    disp_hw=disp_hw,
                                    evaluate=True,
                                    data_dir=args.data_dir,
                                    result_path=args.result_path,
                                    config_path=config_path)
                outstr = '_'+'MIND'+str(int(selected_mind_r))+str(int(selected_mind_d))+'_'+str(int(w_lambda*100))+'lambda_'+str(grid_sp)+'gs1_'+str(disp_hw)+'disp_'+str(use_mask)+'Masks'

                for save_path in save_paths:
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path] = {}
                    eval_json_path = os.path.join(main_save_path,save_path,'metrics'+outstr+'.json')
                    with open(eval_json_path, 'r') as f:
                        eval_data = json.load(f)
                    
                    result_sim_1 = eval_data[args.task_name]['aggregates'][sim_metric_1]['mean']
                    result_sim_1_std = eval_data[args.task_name]['aggregates'][sim_metric_1]['std']
                    result_sim_1_30 = eval_data[args.task_name]['aggregates'][sim_metric_1]['30']
                    if sim_metric_2 != None:
                        result_sim_2 = eval_data[args.task_name]['aggregates'][sim_metric_2]['mean']
                        result_sim_2_std = eval_data[args.task_name]['aggregates'][sim_metric_2]['std']
                        result_sim_2_30 = eval_data[args.task_name]['aggregates'][sim_metric_2]['30']

                    result_smooth = eval_data[args.task_name]['aggregates'][smooth_metric]['mean']
                    result_smooth_std = eval_data[args.task_name]['aggregates'][smooth_metric]['std']
                    result_smooth_30 = eval_data[args.task_name]['aggregates'][smooth_metric]['30']
                    result_time = eval_data[args.task_name]['aggregates']['median_case_time']

                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['time'] = result_time

                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1'] = {}
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth'] = {}

                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['mean'] = result_sim_1
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['std'] = result_sim_1_std
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['30'] = result_sim_1_30
                    
                    if sim_metric_2 != None:
                        experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2'] = {}
                        experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['mean'] = result_sim_2
                        experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['std'] = result_sim_2_std
                        experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['30'] = result_sim_2_30
                    
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['mean'] = result_smooth
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['std'] = result_smooth_std
                    experiments['MIND; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['30'] = result_smooth_30
                

                    with open(main_save_path+'/ablations_'+args.task_name +'.json', 'w') as json_file:
                        json.dump(experiments, json_file)

                if semantic_features:
                    selected_use_mask = False
                    main_for_l2r3_nnUNet.main(task_name=args.task_name, 
                                    mind_r=selected_mind_r,
                                    mind_d=selected_mind_d,
                                    use_mask=selected_use_mask,
                                    lambda_weight=w_lambda,
                                    grid_sp=grid_sp,
                                    disp_hw=disp_hw,
                                    evaluate=True,
                                    data_dir=args.data_dir,
                                    result_path=args.result_path,
                                    config_path=config_path)

                    outstr = '_'+'nnUNet'+'_'+str(int(w_lambda*100))+'lambda_'+str(grid_sp)+'gs1_'+str(disp_hw)+'disp_'+str(selected_use_mask)+'Masks'

                    for save_path in save_paths:
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path] = {}
                        eval_json_path = os.path.join(main_save_path,save_path,'metrics'+outstr+'.json')
                        with open(eval_json_path, 'r') as f:
                            eval_data = json.load(f)
                        
                        result_sim_1 = eval_data[args.task_name]['aggregates'][sim_metric_1]['mean']
                        result_sim_1_std = eval_data[args.task_name]['aggregates'][sim_metric_1]['std']
                        result_sim_1_30 = eval_data[args.task_name]['aggregates'][sim_metric_1]['30']
                        if sim_metric_2 != None:
                            result_sim_2 = eval_data[args.task_name]['aggregates'][sim_metric_2]['mean']
                            result_sim_2_std = eval_data[args.task_name]['aggregates'][sim_metric_2]['std']
                            result_sim_2_30 = eval_data[args.task_name]['aggregates'][sim_metric_2]['30']

                        result_smooth = eval_data[args.task_name]['aggregates'][smooth_metric]['mean']
                        result_smooth_std = eval_data[args.task_name]['aggregates'][smooth_metric]['std']
                        result_smooth_30 = eval_data[args.task_name]['aggregates'][smooth_metric]['30']
                        result_time = eval_data[args.task_name]['aggregates']['median_case_time']

                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['time'] = result_time

                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1'] = {}
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth'] = {}

                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['mean'] = result_sim_1
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['std'] = result_sim_1_std
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim1']['30'] = result_sim_1_30
                        
                        if sim_metric_2 != None:
                            experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2'] = {}
                            experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['mean'] = result_sim_2
                            experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['std'] = result_sim_2_std
                            experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['sim2']['30'] = result_sim_2_30
                        
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['mean'] = result_smooth
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['std'] = result_smooth_std
                        experiments['nnUNet; gs='+str(grid_sp)+'; disp_hwd='+str(disp_hw)+'; lambda='+str(w_lambda)+'; '+save_path]['smooth']['30'] = result_smooth_30
                    

                        with open(main_save_path+'/ablations_'+args.task_name +'.json', 'w') as json_file:
                            json.dump(experiments, json_file)


    # PARAMETER SELECTION BY RANKING    
    with open(main_save_path+'/ablations_'+args.task_name +'.json', 'r') as f:
        data = json.load(f)

    print('provided metrics: ', data['metrics'])
    metric_keys = list(data['metrics'].keys())
    
    if 'DSC' in data['metrics']['sim1']:
        sign_sim1 = 1
    else:
        sign_sim1 = -1

    keys_team = []
    n_rank = 4
    sim1_teams = []
    sim1_30_teams = []
    if ('sim2' in  metric_keys) and (data['metrics']['sim2'] != None):
        sim2_teams = []
        n_rank += 1
        if 'DSC' in data['metrics']['sim2']:
            sign_sim2 = 1
        else:
            sign_sim2 = -1

    smooth_teams = []
    time_teams = []
    for key in data.keys():
        if key.startswith('MIND') or key.startswith('nnUNet'):
            keys_team.append(key)
            sim1_teams.append(data[key]['sim1']['mean'])
            if 'sim2' in  metric_keys:
                sim2_teams.append(data[key]['sim2']['mean'])
            sim1_30_teams.append(data[key]['sim1']['30'])
            smooth_teams.append(data[key]['smooth']['mean'])
            time_teams.append(data[key]['time'])

    N = len(keys_team)
    p_threshold = 0.05

    def scores_better(task_metric):
        better = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                h,p = scipy.stats.ranksums(task_metric[i], task_metric[j])
                if((h>0)&(p<p_threshold)): #sign of h and p-value
                #if((h<0)&(p<p_threshold)): #sign of h and p-value
                    better[i,j] = 1
        scores_task = better.sum(0)
        return scores_task


    def rankscore_avgtie(scores_int):
        N = len(scores_int)
        rankscale = np.linspace(.1,1,N) #our definition
        rankavg = np.zeros((N,2))
        scorerank = np.zeros(N)
        
        #argsort with reverse index
        idx_ = np.argsort(scores_int)
        idx = np.zeros(N).astype('int32')
        idx[idx_] = np.arange(N)
        
        #averaging ties
        for i in range(N):
            rankavg[scores_int[i],0] += rankscale[idx[i]]
            rankavg[scores_int[i],1] += 1
        rankavg = rankavg[:,0]/np.maximum(rankavg[:,1],1e-6)
        for i in range(N):
            scorerank[i] = rankavg[scores_int[i]]
        return scorerank

    def greaters(scores):
        return np.sum(scores.reshape(1,-1)>scores.reshape(-1,1),0)


    rank_all = np.zeros((N,n_rank+1))
    sim1_mean = torch.Tensor(sim1_teams)
    sim1_30 = torch.Tensor(sim1_30_teams)
    if 'sim2' in  metric_keys:
        sim2_mean = torch.Tensor(sim2_teams)
    smooth_all = torch.Tensor(smooth_teams)
    times_all = torch.Tensor(time_teams)
    N39 = N
    P = N

    rank_sim1_mean = np.zeros(N)
    for i in range(50):
        subset = sim1_mean.reshape(N,-1)+.1*np.random.randn(N,N)
        scores = scores_better(sign_sim1*subset)
        rank_sim1_mean += rankscore_avgtie(-scores.astype('int64'))
    rank_sim1_mean = rank_sim1_mean/50
    rank_all[:,0] = rank_sim1_mean
    print('rank sim1_mean done')

    rank_sim1_30 = np.zeros(N)
    for i in range(50):
        subset = sim1_30.reshape(N,-1)+.1*np.random.randn(N,N)
        scores = scores_better(sign_sim1*subset)
        rank_sim1_30 += rankscore_avgtie(-scores.astype('int64'))
    rank_sim1_30 = rank_sim1_30/50
    rank_all[:,1] = rank_sim1_30
    print('rank sim1_30 done')

    rank_smooth = np.zeros(N)
    for i in range(50):
        subset = smooth_all.reshape(N,-1)+.1*np.random.randn(N,N)
        scores = scores_better(-subset)
        rank_smooth += rankscore_avgtie(-scores.astype('int64'))
    rank_smooth/=50
    rank_all[:,2] = rank_smooth
    print('rank smooth done')

    rank_time = np.zeros(N)
    for i in range(50):
        subset = times_all.reshape(N,-1)+.2*np.random.randn(N,N)
        scores = scores_better(-subset)
        rank_time += rankscore_avgtie(-scores.astype('int64'))
    rank_time/=50
    rank_all[:,3] = rank_time
    print('rank time done')

    if 'sim2' in  metric_keys:
        rank_sim2_mean = np.zeros(N)
        for i in range(50):
            subset = sim2_mean.reshape(N,-1)+.1*np.random.randn(N,N)
            scores = scores_better(sign_sim2*subset)
            rank_sim2_mean += rankscore_avgtie(-scores.astype('int64'))
        rank_sim2_mean = rank_sim2_mean/50
        rank_all[:,4] = rank_sim2_mean
        print('rank sim2_mean done')

        rank_all[:,5] = np.power(rank_all[:,0]*np.prod(rank_all[:,:5],axis=1),.2)
        print('WINNER:', np.argmax(rank_all[:,5]), ':' , keys_team[np.argmax(rank_all[:,5])])
        winner_key = keys_team[np.argmax(rank_all[:,5])]
        
    else:
        rank_all[:,4] = np.power(rank_all[:,0]*np.prod(rank_all[:,:4],axis=1),.25)
        print('WINNER:', np.argmax(rank_all[:,4]), ':' , keys_team[np.argmax(rank_all[:,4])])
        winner_key = keys_team[np.argmax(rank_all[:,4])]

    split_key = winner_key.replace(" ", "").split(';')
    if split_key[0] == 'nnUNet':
        select_semantic_features = True
    else:
        select_semantic_features = False
    print('use semantic features = ', select_semantic_features)

    selected_gs = int(split_key[1][-1])
    selected_disp_hwd = int(split_key[2][-1])
    selected_lambda = float(split_key[3][7:])
    selected_niter= int(split_key[4][:2])
    selected_smooth= int(split_key[4][-1])

    if select_semantic_features == False:
        main_for_l2r3_MIND_testset.main(task_name=args.task_name, 
                                        mind_r=selected_mind_r,
                                        mind_d=selected_mind_d,
                                        use_mask=use_mask,
                                        lambda_weight=selected_lambda,
                                        grid_sp=selected_gs,
                                        disp_hw=selected_disp_hwd,
                                        selected_niter=selected_niter,
                                        selected_smooth=selected_smooth,
                                        data_dir=args.data_dir,
                                        result_path=args.result_path)

    if select_semantic_features == True:
        main_for_l2r3_nnUNet_testset.main(task_name=args.task_name, 
                                        mind_r=selected_mind_r,
                                        mind_d=selected_mind_d,
                                        use_mask=use_mask,
                                        lambda_weight=selected_lambda,
                                        grid_sp=selected_gs,
                                        disp_hw=selected_disp_hwd,
                                        selected_niter=selected_niter,
                                        selected_smooth=selected_smooth,
                                        data_dir=args.data_dir,
                                        result_path=args.result_path)


    print('>>> subMISSION completed <<<')
    

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-t","--task_name", dest="task_name", help="provide L2R2022 task name", required=True)
    parser.add_argument("-d",'--data_dir', type=str, default='/share/data_zoe3/grossbroehmer/Learn2Reg2022/Learn2Reg_Dataset_v11/')
    parser.add_argument("-r",'--result_path', type=str, default='./')
    args= parser.parse_args()
    sc_convex_adam(args)