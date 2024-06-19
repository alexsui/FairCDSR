from train_rec import main
import glob
from config.config import get_args
args = get_args()
modes = ["fairness_baseline_Y","fairness_baseline_Y_single"]
training_mode = "evaluation"
domains = ["cross","single"]
main_task = "Y"
ssl = None
data_augmentation = None
time_encode = False
alphas = [0.2,0.4]
dataset = "Movie_lens_main"
folder_list = glob.glob(f"./fairness_dataset/{dataset}/*")
folder_list = [x.split("/")[-1] for x in folder_list]
data_dir = [x for x in folder_list if "sci-fi" in x]
print(data_dir)
warmup_epoch = 100
print("Config of Experiment:")
print(f"Modes: {modes}")
print("training_mode:", training_mode)
print(f"Data: {data_dir}")
num_seeds = 5
for data_idx in range(len(data_dir)):
    data_name = data_dir[data_idx]
    for i, domain in enumerate(domains):
        for seed in range(1, num_seeds+1):
            args.training_mode = training_mode
            args.domain = domain
            args.main_task = main_task
            args.time_encode = time_encode
            args.data_augmentation = data_augmentation     
            args.data_dir = data_name
            args.dataset = dataset
            args.id = f"{modes[i]}_50"
            args.evaluation_model = f"{modes[i]}_50"
            args.seed = seed
            args.warmup_epoch = warmup_epoch
            args.num_cluster = "100,100,100"
            args.ssl = ssl
            main(args)
