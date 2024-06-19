import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from pathlib import Path
import torch

from utils import torch_utils, helper
from utils.GraphMaker import GraphMaker
from model.trainer import CDSRTrainer, Pretrainer
from utils.loader import *
from utils.MoCo_utils import compute_features
from utils.cluster import run_kmeans
from utils.collator import CLDataCollator
from model.item_generator import Generator
from config.config import get_args


def main(args):
    def seed_everything(seed=1111):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("seed set done! seed{}".format(seed))
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    args.num_cluster = [int(n) for n in args.num_cluster.split(',')]
    
    # make opt
    opt = vars(args)
    print("My seed:", opt["seed"])
    seed_everything(opt["seed"])
    model_id = opt["id"]
    folder = opt['save_dir'] + '/'+ str(opt['data_dir'])+ '/' + str(model_id)
    Path(folder).mkdir(parents=True, exist_ok=True)
    model_save_dir = folder + '/' + str(opt['seed'])
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                    header="# test_MRR\ttest_NDCG_10\ttest_HR_10")

    # print model info
    helper.print_config(opt)

    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
    
    if opt["training_mode"] not in ["finetune","joint_learn","evaluation"]:
        raise ValueError("training mode must be finetune, joint_learn or evaluation")
    if opt["training_mode"] in ["joint_learn"] and opt["ssl"] not in ["GMiT","group_CL","both"]:
        raise ValueError("SSL must be GMiT, group_CL or both")
    
    # read number of items
    def read_item(fname):
        with codecs.open(fname, "r", encoding="utf-8") as fr:
            item_num = [int(d.strip()) for d in fr.readlines()[:2]]
        return item_num
    filename = opt["data_dir"]
    opt["source_item_num"], opt["target_item_num"] = read_item(f"./fairness_dataset/{opt['dataset']}/" + filename + "/train.txt")
    opt['itemnum'] = opt["source_item_num"] + opt["target_item_num"] +1
    if opt['data_augmentation'] not in ["item_augmentation","user_generation"] and opt['data_augmentation'] is not None:
        raise ValueError("data augmentation must be item_augmentation or user_generation")
    # load item generator
    if opt['data_augmentation'] == "item_augmentation" or opt['ssl']=="group_CL" or opt['ssl']=="both":
        source_generator = Generator(opt, type='X')
        checkpoint = torch.load(f"./generator_model/{opt['data_dir']}/X/{str(opt['load_pretrain_epoch'])}/model.pt")    
        state_dict = checkpoint['model']
        source_generator.load_state_dict(state_dict)
        target_generator = Generator(opt, type='Y')
        checkpoint = torch.load(f"./generator_model/{opt['data_dir']}/Y/{str(opt['load_pretrain_epoch'])}/model.pt")    
        state_dict = checkpoint['model']
        target_generator.load_state_dict(state_dict)
        mixed_generator = Generator(opt, type='mixed')
        checkpoint = torch.load(f"./generator_model/{opt['data_dir']}/mixed/{str(opt['load_pretrain_epoch'])}/model.pt")    
        state_dict = checkpoint['model']
        mixed_generator.load_state_dict(state_dict)
        print("\033[01;32m Generator loaded! \033[0m")
    # use collator or not for GCL
    if opt['ssl'] in ["group_CL","both"] and opt["substitute_mode"]in ["DGIR","AGIR","random"]: #GAW and Hybrid should warm up for few epochs
        collator = CLDataCollator(opt, eval=-1, mixed_generator=mixed_generator)
    else:
        collator = None
    # build dataloader
    if opt['training_mode'] != "evaluation":
        if opt['data_augmentation']=="item_augmentation":
            train_batch = CustomDataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1, collate_fn  = collator, generator = [source_generator, target_generator, mixed_generator])
        else:
            train_batch = CustomDataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1, collate_fn  = collator)
        valid_batch = CustomDataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 2, collate_fn  = None)
    test_batch = CustomDataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 1,collate_fn  = None)
    print("Data loading done!")    
    # model
    trainer = CDSRTrainer(opt)
    if opt['training_mode']=="evaluation":
        if opt['evaluation_model'] is None:
            raise ValueError("evaluation model is not specified!") 
        if opt["main_task"]=="X":
            evaluation_path ="models/" + opt['data_dir'] + f"/{opt['evaluation_model']}/{str(opt['seed'])}/X_model.pt" 
        elif opt["main_task"]=="Y":    
            evaluation_path ="models/" + opt['data_dir'] + f"/{opt['evaluation_model']}/{str(opt['seed'])}/Y_model.pt"
        print("evaluation_path",evaluation_path)
        if os.path.exists(evaluation_path):
            print("\033[01;32m Loading evaluation model from {}... \033[0m\n".format(evaluation_path))
            trainer.load(evaluation_path)
            print("\033[01;32m Loading evaluation model done! \033[0m\n")
        else:
            raise ValueError("evaluation model does not exist!")
        print("\033[01;34m Start evaluation... \033[0m\n")
        best_Y_test, best_Y_test_male,best_Y_test_female = trainer.evaluate(test_batch, file_logger)
        return best_Y_test, best_Y_test_male,best_Y_test_female
    else:
        print("\033[01;32m Model training from scratch... \033[0m\n")
    if opt['training_mode']=="joint_learn":
        print("\033[01;34m Start joint learning... \033[0m\n")
    if opt['data_augmentation']=="item_augmentation" or opt['ssl']in ["group_CL","both"]:
        trainer.generator = [source_generator, target_generator, mixed_generator]
    best_test,best_test_male,best_test_female =  trainer.train(opt['num_epoch'], train_batch, valid_batch, test_batch, file_logger)
    return best_test,best_test_male,best_test_female
if __name__ == '__main__':
    args  = get_args()
    main(args)