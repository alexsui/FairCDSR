import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)
import glob
import pandas as pd
from pathlib import Path
from train_rec import main
from config.config import get_args
args = get_args()
substitute_modes = ["hybrid","hybrid","hybrid","hybrid"]
substitute_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
dataset = "Movie_lens_main"
data_dirs = glob.glob(f"./fairness_dataset/{dataset}/*")
data_dirs = [x.split("/")[-1] for x in data_dirs] 
for sub_mode, data_dir in zip(substitute_modes,data_dirs):
    columns_name = ["substitute_ratio","seed",
                    "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                    "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                    "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                    ]
    res_df = pd.DataFrame(columns=columns_name)
    for i in range(5):
        for ratio in substitute_ratios:
            args.substitute_mode = sub_mode
            args.substitute_ratio = ratio
            args.data_dir = data_dir
            args.dataset = dataset
            args.seed = i
            args.num_epoch = 200
            args.ssl = "group_CL"
            args.training_mode = "joint_learn"
            args.num_cluster = "100,100,4"
            args.id = f"RQ1_ratio{ratio}"
            best_Y_test,best_Y_test_male,best_Y_test_female = main(args)
            df = pd.DataFrame([[ratio,i]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
            res_df = pd.concat([res_df,df],axis=0)
    Path(f"./RQ1/substitute_ratio/substitution_ratio_res/").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(f"./RQ1/substitute_ratio/substitution_ratio_res/{data_dir}.csv")