import argparse
import time
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from train_rec import main
import ipdb
import pandas as pd
from pathlib import Path
from config.config import get_args
args = get_args()

cluster_numbers = ["700,700,700","500,500,500","500,500,500","400,400,400"]
data_dirs = ["sci-fi_thriller","action_comedy","comedy_drama","sci-fi_comedy"]
cluster_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
dataset = "Movie_lens_main"
for data_dir, cluster_number in zip(data_dirs,cluster_numbers):
    columns_name = ["is_baseline","seed","cluster_number","cluster_ratio",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                    ]
    res_df = pd.DataFrame(columns=columns_name)
    for i in range(0,5):
        for cluster_ratio in cluster_ratios:
            args.cluster_ratio = cluster_ratio
            args.num_cluster = cluster_number
            args.data_dir = data_dir
            args.dataset = dataset
            args.seed = i
            args.num_epoch = 200
            args.cluster_mode = 'separate'
            args.ssl = "GMiT"
            args.training_mode = "joint_learn"
            args.id = f"RQ2_cluster_ratio{cluster_ratio}"
            best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
            is_baseline = False
            df = pd.DataFrame([[is_baseline,i,cluster_number,cluster_ratio]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
            res_df = pd.concat([res_df,df],axis=0)
        args.data_dir = data_dir
        args.dataset = dataset
        args.seed = i
        args.num_epoch = 200
        args.ssl = None
        args.training_mode = "finetune"
        args.id = f"RQ2_cluster_ratio_baseline"
        args.num_cluster = "1,1,1"
        best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
        is_baseline = True
        df = pd.DataFrame([[is_baseline,i,None,None]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
        res_df = pd.concat([res_df,df],axis=0)
    Path(f"./RQ2/cluster_ratio/cluster_ratio_res/").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(f"./RQ2/cluster_ratio/cluster_ratio_res/{data_dir}.csv")