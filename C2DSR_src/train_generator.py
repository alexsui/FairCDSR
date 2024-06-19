import torch
import os
from utils import helper
from model.trainer import GTrainer
from utils.loader import *
from utils.collator import GDataCollator
from pathlib import Path
import argparse
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
    if isinstance(args.num_cluster, str):
        args.num_cluster = [int(n) for n in args.num_cluster.split(',')]
    # make opt
    opt = vars(args)
    print("My seed:", opt["seed"])
    seed_everything(opt["seed"])
    model_id = opt["id"]
    folder = opt['save_dir'] + '/'+ str(opt['data_dir'])+ '/' + str(model_id)
    Path(folder).mkdir(parents=True, exist_ok=True)
    model_save_dir = folder + '/' 
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    helper.print_config(opt)
    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
    collator = GDataCollator(opt)
    train_batch = GDataLoader(opt['data_dir'], opt['batch_size'], opt,eval = -1, collate_fn = collator)
    val_batch = GDataLoader(opt['data_dir'], opt['batch_size'], opt,eval = 1, collate_fn = collator)
    trainer = GTrainer(opt)
    trainer.train(train_batch, val_batch)
    return 