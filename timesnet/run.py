import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
import torch.backends
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
from src.utils.utils import load_args, load_gpu

if __name__ == '__main__':
    args = load_args("args.json")
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    fix_seed = args.seed
    
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass

    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    args = load_args("args.json")

    flag = "train"

    args = load_gpu(args)

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'classification':
        Exp = Exp_Classification

    df = pd.read_csv(os.path.join(args.root_path,
                                args.data_path))

    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(df, args, flag)  # set experiments
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_po{args.pira_offset}_ms{args.min_seq_len}_dm{args.d_model}_do{args.dropout}_el{args.e_layers}_df{args.d_ff}_lr{args.learning_rate}_bs{args.batch_size}_sw{args.seq_len}_pw{args.pred_window}_cw{args.class_weight[0]}_{args.class_weight[1]}"

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, args)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
