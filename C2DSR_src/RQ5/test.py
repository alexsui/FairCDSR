import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) #/mnt/samuel/C2DSR_fairness/C2DSR_src/RQ4
parent_dir = os.path.dirname(current_dir) #/mnt/samuel/C2DSR_fairness/C2DSR_src
sys.path.insert(0, parent_dir)
from train_rec import main
import pandas as pd
from pathlib import Path
from config.config import get_args

args  = get_args()
training_settings = ['single','C2DSR']
dataset = "age_test"
ages = [1,18,25,35,45,50,56]
data_dirs = ["action_comedy","comedy_drama","sci-fi_thriller","drama_sci-fi"]
columns_name = ["scenario","age","training_setting","seed","alpha","substitute_ratio",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
               ]
res_df = pd.DataFrame(columns=columns_name)
for  data_dir in data_dirs:
    for age in ages:
        for training_setting in training_settings:
            for seed in range(0,1):
                args.data_dir = data_dir+"/"+str(age)
                args.dataset = f"{dataset}"
                args.id = f"RQ6_{training_setting}"  
                args.seed = seed
                args.num_epoch = 200
                if training_setting == "single":
                    args.domain = "single"
                    args.training_mode = "finetune"
                    args.ssl = None
                    args.data_augmentation = None
                    args.num_cluster = "100,100,200"
                    best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                    df = pd.DataFrame([[data_dir,str(age), training_setting, seed,0,0]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                    res_df = pd.concat([res_df, df],axis=0)
                elif training_setting == "C2DSR":
                    args.domain = "cross"
                    args.training_mode = "finetune"
                    args.ssl = None
                    args.data_augmentation = None
                    args.num_cluster = "100,100,200"
                    best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
                    df = pd.DataFrame([[data_dir, str(age), training_setting, seed,0,0]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                    res_df = pd.concat([res_df, df],axis=0)
Path(f"./RQ6/age_test/result").mkdir(parents=True, exist_ok=True)
res_df.to_csv(f"./RQ6/age_test/result/res.csv")
