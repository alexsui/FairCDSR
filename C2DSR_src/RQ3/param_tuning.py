import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import optuna
import glob
import glob
from train_rec import main
from config.config import get_args


def objective(trial, data_dir):
    #item augmentation
    alpha = trial.suggest_float("alpha", 0.3, 0.8, step=0.1)
    #GMiT
    topk_cluster = trial.suggest_categorical("topk_cluster", [3, 5, 7])
    num_clusters_option = trial.suggest_categorical("num_clusters_option", ["100,100,100","200,200,200","300,300,300","400,400,400","500,500,500"])
    # warmup_epoch = trial.suggest_categorical("warmup_epoch",[1,10])
    #group CL
    substitute_ratio = trial.suggest_float("substitute_ratio", 0.3, 0.9, step=0.1)
    # loss weight
    lambda_0 = trial.suggest_float("lambda_0", 0.2, 1, step =0.1)  # Adjust the range as needed
    lambda_1 = trial.suggest_float("lambda_1", 0.2, 1, step =0.1)  # Adjust the range as needed
    args.ssl = "both"
    args.main_task = "Y"
    args.domain = "cross"
    args.seed = 2024
    args.training_mode = "joint_learn"
    args.data_augmentation = "item_augmentation"
    args.dataset = "Movie_lens_main"
    args.substitute_mode = "AGIR"
    args.id = f"alpha{round(alpha,3)}_topk{topk_cluster}_num_cluster{num_clusters_option}_substitute{round(substitute_ratio,3)}_lambda_{round(lambda_0,3)}_{round(lambda_1,3)}"
    args.data_dir = data_dir
    args.substitute_ratio = round(substitute_ratio,3)
    args.topk_cluster = topk_cluster
    args.num_cluster = num_clusters_option
    args.alpha = round(alpha,3)
    args.lambda_ = [round(lambda_0,3), round(lambda_1,3)]
    best_Y_test, best_Y_test_male, best_Y_test_female  = main(args)
    trial.set_user_attr(key="best_Y_test", value=best_Y_test)
    trial.set_user_attr(key="best_Y_test_male", value=best_Y_test_male)
    trial.set_user_attr(key="best_Y_test_female", value=best_Y_test_female)
    return best_Y_test_male[2] - best_Y_test_female[2], best_Y_test[2] #NDCG@5
def callback(study, trial):
    for best_trial in study.best_trials:
        if best_trial.number == trial.number:
            # Update the study's user attributes with the best trial's additional info
            study.set_user_attr('best_Y_test', trial.user_attrs['best_Y_test'])
            study.set_user_attr('best_Y_test_male', trial.user_attrs['best_Y_test_male'])
            study.set_user_attr('best_Y_test_female', trial.user_attrs['best_Y_test_female'])
            break
args = get_args()
folder = "Movie_lens_main"
folder_list = glob.glob(f"./fairness_dataset/{folder}/*")
folder_list = [x.split("/")[-1] for x in folder_list]
folder_list = ["comedy_drama","sci-fi_thriller"]
for folder in folder_list:
    study = optuna.create_study(directions=['minimize','maximize'])  # maximize: overall test_NDCG_5, minimize: male_NDCG_5 - female_NDCG_5
    study.optimize(lambda trial: objective(trial, folder), n_trials=100, callbacks=[callback])
    
    if study.best_trials:
        for best_trial in study.best_trials:
            print(f"{folder} - Trial#{best_trial.number} - best_params :", best_trial.params)
            print(f"{folder} - Trial#{best_trial.number} - best_value :", best_trial.values)
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test: {best_trial.user_attrs['best_Y_test']}")
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test_male: {best_trial.user_attrs['best_Y_test_male']}")
            print(f"{folder} - Trial#{best_trial.number} - best_Y_test_female: {best_trial.user_attrs['best_Y_test_female']}")
        file_path = f'param_tune_result/{folder}.txt'
        with open(file_path, 'w') as file:
            for best_trial in study.best_trials:
                file.write(f"{folder} - Trial#{best_trial.number} - best_params :{best_trial.params}\n")
                file.write(f"{folder} - - Trial#{best_trial.number} - best_value :{best_trial.values}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test: {best_trial.user_attrs['best_Y_test']}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test_male: {best_trial.user_attrs['best_Y_test_male']}\n")
                file.write(f"{folder} - Trial#{best_trial.number} - best_Y_test_female: {best_trial.user_attrs['best_Y_test_female']}\n")              
                file.write("-"*50 + "\n")
    