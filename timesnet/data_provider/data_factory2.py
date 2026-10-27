from timesnet.data_provider.data_loader import Dataset_Custom
from timesnet.data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from functools import partial
import random
import numpy as np
import torch

data_dict = {
    'custom': Dataset_Custom,
}

def my_collate_fn(batch, max_len, mode):
    # Aquí llamas a tu collate_fn original con los parámetros que necesitas
    return collate_fn(batch, max_len=max_len, mode=mode)


def worker_init_fn(worker_id, base_seed):  # <--- Asegúrate de que tenga base_seed aquí
    import random
    import numpy as np
    import torch
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def data_provider(args, flag, dataset_x=None, dataset_y=None):
    Data = data_dict["custom"]
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size

    collate = partial(my_collate_fn, max_len=args.seq_len, mode=args.mode)

    if args.task_name == 'classification':
        data_set = Data(
            args=args,
            flag=flag,
            dataset_x=dataset_x,
            dataset_y=dataset_y,
        )

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        # 1. ESTA ES LA FUNCIÓN "EMPAQUETADA" QUE DEBES USAR
        worker_init = partial(worker_init_fn, base_seed=args.seed)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collate,
            # 2. AQUÍ DEBE IR 'worker_init' (la versión con partial), NO 'worker_init_fn'
            worker_init_fn=worker_init,
            generator=generator,
        )
        return data_set, data_loader