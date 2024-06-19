import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) #/mnt/samuel/C2DSR_fairness/C2DSR_src/RQ2/cluster_mode
parent_dir = os.path.dirname(os.path.dirname(current_dir)) #/mnt/samuel/C2DSR_fairness/C2DSR_src
sys.path.insert(0, parent_dir)
from train_rec import main

import glob
import pandas as pd
from pathlib import Path
from train_rec import main
from config.config import get_args
args = get_args()
cluster_modes = ["separate","joint"] 
cluster_numbers = ["25,25,50","50,50,100","75,75,150","100,100,200","125,125,250","150,150,300","175,175,350","200,200,400","225,225,450","250,250,500"]
dataset = "Movie_lens_main"
data_dirs = glob.glob(f"./fairness_dataset/{dataset}/*")
data_dirs = [x.split("/")[-1] for x in data_dirs] 
for data_dir in data_dirs:
    columns_name = ["is_baseline","seed","cluster_number","cluster_mode",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                    ]
    res_df = pd.DataFrame(columns=columns_name)
    for i in range(0,5):
        for cluster_mode in cluster_modes:
            for cluster_number in cluster_numbers:
                args.cluster_mode = cluster_mode
                args.num_cluster = cluster_number
                args.data_dir = data_dir
                args.dataset = dataset
                args.seed = i
                args.num_epoch = 200
                args.ssl = "GMiT"
                args.training_mode = "joint_learn"
                args.id = f"RQ2_cluster_mode{cluster_mode}_cluster_number{cluster_number}"
                best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                is_baseline = False
                df = pd.DataFrame([[is_baseline,i,cluster_number,cluster_mode]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                res_df = pd.concat([res_df,df],axis=0)
        # baseline for every seed
        args.data_dir = data_dir
        args.dataset = dataset
        args.seed = i
        args.num_epoch = 200
        args.ssl = None
        args.training_mode = "finetune"
        args.id = f"RQ2_cluster_mode_baseline"
        args.num_cluster = "1,1,1"
        best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
        is_baseline = True
        df = pd.DataFrame([[is_baseline,i,None,None]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
        res_df = pd.concat([res_df,df],axis=0)
    Path(f"./RQ2/cluster_mode/cluster_mode_res").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(f"./RQ2/cluster_mode/cluster_mode_res/{data_dir}.csv")
    