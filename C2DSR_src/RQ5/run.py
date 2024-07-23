import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) #/mnt/samuel/C2DSR_fairness/C2DSR_src/RQ4
parent_dir = os.path.dirname(current_dir) #/mnt/samuel/C2DSR_fairness/C2DSR_src
sys.path.insert(0, parent_dir)
import pandas as pd
from pathlib import Path
import glob
from train_rec import main
from config.config import get_args

args  = get_args()
training_settings = ['FairCDSR',"single","C2DSR"]
dataset = "RQ5_dataset/1and45age_dataset"
alphas = [0.5]
substitute_ratios = [0.5,0.6,0.7,0.8]
data_dirs = glob.glob(f"./fairness_dataset/{dataset}/*")
data_dirs = [x.split("/")[-1] for x in data_dirs] 
columns_name = ["scenario","training_setting","seed","alpha","substitute_ratio",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                ]
res_df = pd.DataFrame(columns=columns_name)
for  data_dir in data_dirs:
    for training_setting in training_settings:
        for seed in range(0,1):
            args.data_dir = data_dir
            args.dataset = f"{dataset}"
            args.id = f"RQ5_{training_setting}"  
            args.seed = seed
            args.num_epoch = 200
            if training_setting == "single":
                args.domain = "single"
                args.training_mode = "finetune"
                args.ssl = None
                args.data_augmentation = None
                args.num_cluster = "100,100,200"
                best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                df = pd.DataFrame([[data_dir, training_setting, seed,0,0]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                res_df = pd.concat([res_df, df],axis=0)
            elif training_setting == "C2DSR":
                args.domain = "cross"
                args.training_mode = "finetune"
                args.ssl = None
                args.data_augmentation = None
                args.num_cluster = "100,100,200"
                best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                df = pd.DataFrame([[data_dir, training_setting, seed,0,0]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                res_df = pd.concat([res_df, df],axis=0)
            else:
                for alpha in alphas:
                    for substitute_ratio in substitute_ratios:
                        args.alpha = alpha
                        args.substitute_ratio = substitute_ratio
                        args.id = f"RQ5_{training_setting}_alpha{alpha}_sub{substitute_ratio}"
                        args.domain = "cross"
                        args.training_mode = "joint_learn"
                        args.ssl = "both"
                        args.cluster_mode = "separate"
                        args.num_cluster = "250,250,500"
                        args.data_augmentation = "item_augmentation"
                        best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                        df = pd.DataFrame([[data_dir, training_setting, seed,alpha,substitute_ratio]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                        res_df = pd.concat([res_df,df],axis=0)
Path(f"./RQ5/1and45/result").mkdir(parents=True, exist_ok=True)
res_df.to_csv(f"./RQ5/1and45/result/res.csv")
