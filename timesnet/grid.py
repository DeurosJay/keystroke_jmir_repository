import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from itertools import product
from exp.exp_classification import Exp_Classification
import torch
from src.utils.utils import load_args, load_gpu
import pandas as pd
import torch.multiprocessing as mp

import random
import numpy as np

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
if __name__ == "__main__":
    
    mp.freeze_support()  # opcional, pero recomendado en Windows

    args = load_args("args.json")

    flag = 'train'

    args = load_gpu(args)
    
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

    param_grid = {
#<<<<<<< HEAD
        #"pira_offset": [30, 40],
        #"min_seq_len": [40, 50, 60],
        # "dropout": [0.2, 0.3],
        #"learning_rate": [0.0001, 0.0005, 0.001],
        "class_weight": [[1.0, 5.0], [1.0, 6.0], [1.0, 5.0]]
#=======
#        "pira_offset": [30, 40],
 #       "min_seq_len": [40, 50, 60],
  #      "dropout": [0.2, 0.3],
   #     "learning_rate": [0.0001, 0.0005, 0.001],
    #    "class_weight": [[1.0, 8.0], [1.0, 6.0]]
#>>>>>>> be9ed293cea1357f7226549516906676439b85c0
    }

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    for i, combo in enumerate(param_combinations):
        print(f">>>>> Configuración {i+1} {combo} <<<<<")
        print(f"\n>>>>> Ejecutando configuración {i+1}/{len(param_combinations)} <<<<<")
        combo_dict = dict(zip(param_names, combo))

        # Sobrescribimos Args
        for key, val in combo_dict.items():
            setattr(args, key, val)

        # Cambiar nombre del experimento para que no sobreescriba checkpoints
        args.des = f'combo{i}'
        
        df = pd.read_csv(os.path.join(args.root_path,
                                    args.data_path))
        
        # Ejecutar experimento
        Exp = Exp_Classification
        exp = Exp(df, args, flag)
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_po{args.pira_offset}_ms{args.min_seq_len}_dm{args.d_model}_do{args.dropout}_el{args.e_layers}_df{args.d_ff}_lr{args.learning_rate}_bs{args.batch_size}_sw{args.seq_len}_pw{args.pred_window}_cw{args.class_weight[0]}_{args.class_weight[1]}"

        print(f">>>>> Entrenando configuración: {setting}")
        exp.train(setting, args)
        print(f">>>>> Testeando configuración: {setting}")
        exp.test(setting)

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
