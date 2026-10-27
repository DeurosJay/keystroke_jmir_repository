import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
import torch.backends
import random
import numpy as np
import pandas as pd

from src.utils.utils import clean_demografico, merge_dataframes, load_gpu, load_args, load_model_scaler
from src.database.mongo import get_df_key
from utils.print_args import print_args
from exp.exp_classification import Exp_Classification

# Hago preprocessing
df = pd.read_excel('../data/demografico.xlsx')
df_cleaned = clean_demografico(df)
keys = df_cleaned['key'].unique()
print(keys)

# Descargo dataset por paciente y creo df
for key in keys:
    df = get_df_key(key)
    df_final = merge_dataframes(df, df_cleaned)
    print(f"Processed key: {key}")

# Ejecuto run.

if __name__ == '__main__':
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

    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(df_final, args, flag)  # set experiments
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_po{args.pira_offset}_ms{args.min_seq_len}_dm{args.d_model}_do{args.dropout}_el{args.e_layers}_df{args.d_ff}_lr{args.learning_rate}_bs{args.batch_size}_sw{args.seq_len}_pw{args.pred_window}_cw{args.class_weight[0]}_{args.class_weight[1]}"

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, args)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
