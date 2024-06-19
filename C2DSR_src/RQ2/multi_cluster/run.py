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
cluster_numbers = ["700,700,700"]
data_dirs = ["drama_sci-fi"]
cluster_ratios = [0.5]
topk_clusters = [1,3,5,7,9,11,13,15]

dataset = "Movie_lens_main"
for data_dir, cluster_number,cluster_ratio in zip(data_dirs,cluster_numbers,cluster_ratios):
    columns_name = ["seed","topk_cluster",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                    ]
    res_df = pd.DataFrame(columns=columns_name)
    for i in range(0,5):
        for topk_cluster in topk_clusters:
            args.cluster_ratio = cluster_ratio
            args.num_cluster = cluster_number
            args.topk_cluster = topk_cluster
            args.data_dir = data_dir
            args.dataset = dataset
            args.seed = i
            args.num_epoch = 200
            args.cluster_mode = 'separate'
            args.ssl = "GMiT"
            args.training_mode = "joint_learn"
            args.id = f"RQ2_topk_cluster{topk_cluster}"
            best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
            df = pd.DataFrame([[i,topk_cluster]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
            res_df = pd.concat([res_df,df],axis=0)
    Path(f"./RQ2/multi_cluster/multi_cluster_res/").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(f"./RQ2/multi_cluster/multi_cluster_res/{data_dir}.csv")