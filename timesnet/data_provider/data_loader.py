import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from timesnet.data_provider.uea import Normalizer
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args,flag='train', features='S',
                 target='OT', scale=True, timeenc=0, freq='h', dataset_x=None, dataset_y=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.batch_size = args.batch_size
        
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.pred_window = args.pred_window
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = args.root_path
        self.data_path = args.data_path
        self.data_x = dataset_x
        self.data_y = dataset_y
        # [FIX D3] Se elimina el método muerto __read_data__: hacía un split
        # estratificado A NIVEL DE VENTANA (fuga de paciente), con otro etiquetado
        # y solo 12 features (vs. las 20 reales). El dataset se construye en
        # create_dataset (split por paciente, normalización solo-train) y se inyecta
        # vía dataset_x/dataset_y. Mantenerlo era un riesgo si se reactivaba.

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]