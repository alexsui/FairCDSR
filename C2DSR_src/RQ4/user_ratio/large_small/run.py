import argparse
import time
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) #/mnt/samuel/C2DSR_fairness/C2DSR_src/RQ4/user_ratio/manual
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) #/mnt/samuel/C2DSR_fairness/C2DSR_src
sys.path.insert(0, parent_dir)
from train_rec import main
import glob
import ipdb
import pandas as pd
from pathlib import Path
parser = argparse.ArgumentParser()
# dataset part
parser.add_argument(
    "--data_dir",
    type=str,
    default="Food-Kitchen",
    help="Movie-Book, Entertainment-Education",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="Movie_lens_time",
    help="Movie-Book, Entertainment-Education",
)
# model part
parser.add_argument("--model", type=str, default="C2DSR", help="model name")
parser.add_argument("--hidden_units", type=int, default=128, help="lantent dim.")
parser.add_argument("--num_blocks", type=int, default=2, help="lantent dim.")
parser.add_argument("--num_heads", type=int, default=1, help="lantent dim.")
parser.add_argument("--GNN", type=int, default=1, help="GNN depth.")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adam",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument('--param_group', type = bool, default=False, help='param group')
parser.add_argument(
    "--lr", type=float, default=0.001, help="Applies to sgd and adagrad."
)
parser.add_argument(
    "--lr_decay", type=float, default=1, help="Learning rate decay rate."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument(
    "--decay_epoch", type=int, default=5, help="Decay learning rate after this epoch."
)
parser.add_argument(
    "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
)
parser.add_argument("--leakey", type=float, default=0.1)
parser.add_argument("--maxlen", type=int, default=50)
parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
parser.add_argument(
    "--cuda",
    type=bool,
    default=torch.cuda.is_available(),
    help="Enables CUDA training.",
)
parser.add_argument("--lambda", type=float, default=0.7)

# train part
parser.add_argument(
    "--num_epoch", type=int, default=200, help="Number of total training epochs."
)
parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
parser.add_argument(
    "--log_step", type=int, default=200, help="Print log every k steps."
)
parser.add_argument(
    "--log", type=str, default="log.txt", help="Write training log to file."
)
parser.add_argument(
    "--save_epoch", type=int, default=100, help="Save model checkpoints every k epochs."
)
parser.add_argument(
    "--save_dir", type=str, default="./saved_models", help="Root dir for saving models."
)
parser.add_argument(
    "--id", type=str, default=00, help="Model ID under which to save models."
)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument(
    "--load",
    dest="load",
    action="store_true",
    default=False,
    help="Load pretrained model.",
)
parser.add_argument("--model_file", type=str, help="Filename of the pretrained model.")
parser.add_argument(
    "--info", type=str, default="", help="Optional info for the experiment."
)
parser.add_argument("--undebug", action="store_false", default=True)

# data augmentation
parser.add_argument("--augment_type", type=str, default="dropout", help="augment type")
parser.add_argument("--crop_prob", type=float, default=0.7, help="crop probability")
parser.add_argument('--mask_prob', type=float, default=0.2, help='mask probability')
# time ssl
parser.add_argument("--window_size", type=int, default=3, help="window size for ssl")
parser.add_argument("--temp", type=float, default=0.05, help="temperature for ssl")
parser.add_argument(
    "--ssl",
    type=str,
    default="proto_CL",
    help="[time_CL, augmentation_based_CL, no_ssl, proto_CL]",
)

# early stop
parser.add_argument("--pretrain_patience", type=int, default=1000, help="early stop counter")
parser.add_argument("--finetune_patience", type=int, default=5, help="early stop counter")

parser.add_argument("--pooling", type=str, default="ave", help="pooling method")
parser.add_argument("--is_pooling", type=bool, default=True, help="pooling or not")

# MoCo
parser.add_argument(
    "--r", type=int, default=2048, help="queue size/negative sample"
)  # warning : r must be divisible by batch_size
parser.add_argument(
    "--m", type=float, default=0.999, help="momentum update ratio for moco"
)

parser.add_argument("--mlp", type=bool, default=True, help="use pojector or not")
# args = parser.parse_args()

parser.add_argument('--cross_weight',type=float,default=0.001 ,help="cross domain weight for proto CL")
parser.add_argument('--num_proto_neg',type=int,default= 1280 ,help="intra domain weight for proto CL")

#pretrain
parser.add_argument('--training_mode',default= "finetune", type = str, help='["pretrain","joint_pretrain","finetune","joint_learn"]')
parser.add_argument('--pretrain_epoch',type=int,default= 70 ,help="pretrain epoch")
# parser.add_argument('--joint_learn',default= False , action='store_true', help="ssl + main task")
# parser.add_argument('--joint_pretrain', default= False, action='store_true', help="ssl + two encoder  loss task")

parser.add_argument('--load_pretrain_epoch',type=int,default= 20 ,help="pretrain epoch")
parser.add_argument('--pretrain_model',type=str,default= None ,help="pretrain or not")

parser.add_argument('--time_encode',type=bool,default= False ,help="=time encoding or not")
parser.add_argument('--time_embed', type=int, default= 128, help='time dim.')
parser.add_argument('--speedup_scale', type=list, default = [0.3, 0.5] , help='speedup_scale')
parser.add_argument('--slowdown_scale', type=int, default= 2, help='slowdown_scale')
parser.add_argument('--time_transformation', type=str, default="speedup", help="[speedup,slowdown]")

parser.add_argument('--valid_epoch', type=int, default= 3, help='valid_epoch')
parser.add_argument('--mixed_included',type=bool, default= False ,help="mixed included or not")
parser.add_argument('--main_task',type=str, default="Y" ,help="[dual, X, Y]")


parser.add_argument('--data_augmentation',type=str,default= None ,help="[item_generation, user_generation]")
parser.add_argument('--evaluation_model',type=str,default= None ,help="evaluation model")
parser.add_argument('--domain', type=str,default= "cross" ,help="target only or cross domain")
parser.add_argument('--topk', type=int,default= 10 ,help="topk item recommendation")
# item generation
parser.add_argument('--generate_type',type=str,default= "X" ,help="[X,Y,mixed]")
parser.add_argument('--generate_num',type=int,default= 5 ,help="number of item to generate")
parser.add_argument('--alpha',type=float,default= 0.4 ,help="insertion ration for DGSA")
# nonoverlap user augmentation
parser.add_argument('--augment_size',type=int,default= 30 ,help="nonoverlap_augment size")
#interest clustering
parser.add_argument('--topk_cluster',type=str,default= 7 ,help="number of multi-view cluster")
parser.add_argument('--num_cluster', type=str, default= '100,100,200' ,help="number of clusters for kmeans")
parser.add_argument('--cluster_mode',type=str,default= "separate" ,help="separate or joint")
parser.add_argument('--warmup_epoch', type=int, default= 0 ,help="warmup epoch for cluster")
parser.add_argument('--cluster_ratio',type=float, default= 0.5 ,help="cluster ratio")
#group CL
parser.add_argument('--substitute_ratio',type=float, default= 0.6 ,help="substitute ratio")
parser.add_argument('--substitute_mode',type=str, default= "hybrid" ,help="IR, attention_weight")
# loss weight
parser.add_argument('--lambda_',nargs='*', default= [1,1] ,help="loss weight")

parser.add_argument('--C2DSR',type=bool,default= False ,help="if use C2DSR")
parser.add_argument('--RQ4_user_ratio',type=float,default= 1.0  ,help="user ratio")
parser.add_argument('--RQ4',type=bool,default= False  ,help="if train RQ4")
args, unknown = parser.parse_known_args()

#training
#training_mode,domain,ssl
training_settings = ['single','C2DSR','our']
dataset = "RQ4_dataset"
data_modes = ['large',"small"]
for data_mode in data_modes:
    data_dirs = glob.glob(f"./fairness_dataset/{dataset}/user_ratio/{data_mode}/*")
    data_dirs = [x.split("/")[-1] for x in data_dirs] 
    columns_name = ["data_dir","training_setting","seed","data_mode",
                        "test_Y_MRR", "test_Y_NDCG_5", "test_Y_NDCG_10", "test_Y_HR_5", "test_Y_HR_10",
                        "test_Y_MRR_male", "test_Y_NDCG_5_male", "test_Y_NDCG_10_male", "test_Y_HR_5_male", "test_Y_HR_10_male",
                        "test_Y_MRR_female", "test_Y_NDCG_5_female", "test_Y_NDCG_10_female", "test_Y_HR_5_female", "test_Y_HR_10_female"
                        ]
    res_df = pd.DataFrame(columns=columns_name)
    for data_dir in data_dirs:
        for training_setting in training_settings:
            for i in range(0,3):
                args.data_dir = data_dir
                args.dataset = f"{dataset}/user_ratio/{data_mode}"
                args.id = f"RQ4_{training_setting}_{data_mode}"  
                args.seed = i
                args.num_epoch = 200
                if training_setting == "single":
                    args.domain = "single"
                    args.training_mode = "finetune"
                    args.ssl = None
                    args.data_augmentation = None
                    args.num_cluster = "100,100,200"
                elif training_setting == "C2DSR":
                    args.domain = "cross"
                    args.training_mode = "finetune"
                    args.ssl = None
                    args.data_augmentation = None
                    args.num_cluster = "100,100,200"
                elif training_setting == "our":
                    args.domain = "cross"
                    args.training_mode = "joint_learn"
                    args.ssl = "both"
                    args.cluster_mode = "separate"
                    args.num_cluster = "250,250,500"
                    args.data_augmentation = "item_augmentation"
                else:
                    raise ValueError(f"training_setting {training_setting} not found")
                best_Y_test, best_Y_test_male, best_Y_test_female = main(args)
                df = pd.DataFrame([[data_dir, training_setting, i,data_mode]+best_Y_test+best_Y_test_male+best_Y_test_female],columns = columns_name)
                res_df = pd.concat([res_df,df],axis=0)
        
    Path(f"./RQ4/user_ratio/large_small/result").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(f"./RQ4/user_ratio/large_small/result/{data_mode}.csv")
    