from train_generator import main
import glob
from config.config import get_args
args = get_args()
dataset = "Movie_lens_main"
folder_list = glob.glob(f"./fairness_dataset/{dataset}/*")
data_dir = [x.split("/")[-1] for x in folder_list]
print(data_dir)
generate_types = ["X","Y","mixed"]
print("Config of Experiment:")
num_seeds = 1
for data_idx in range(len(data_dir)):
    data_name = data_dir[data_idx]
    for generate_type in generate_types:            
        args.data_dir = data_name
        args.generate_type = generate_type
        args.id = f"{generate_type}"
        args.dataset = dataset
        main(args)
