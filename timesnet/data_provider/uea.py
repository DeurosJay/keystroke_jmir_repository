import os
import numpy as np
import pandas as pd
import torch

def collate_fn(data, max_len=None, mode='normal'):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Convertir a np.float32 si no lo están
    features = [np.asarray(f, dtype=np.float32) for f in features]
    lengths = [f.shape[0] for f in features]
    feat_dim = features[0].shape[1]
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)

    if mode == 'window':
        # Construye padding_masks manualmente
        padding_masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
        for i, f in enumerate(features):
            end = min(f.shape[0], max_len)
            X[i, :end, :] = torch.from_numpy(f[:end, :])
            padding_masks[i, :end] = 1.0
    elif mode == 'normal':
        # Usa la función padding_mask externa
        for i, f in enumerate(features):
            end = min(f.shape[0], max_len)
            X[i, :end, :] = torch.from_numpy(f[:end, :])
        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    else:
        raise ValueError(f"Modo desconocido: {mode}. Usa 'normal' o 'window'.")

    # Crear labels tensor
    targets = torch.tensor(labels, dtype=torch.long)

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class Normalizer(object):
    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """
        self.norm_type = norm_type
        self.mean = mean.values if isinstance(mean, pd.Series) else mean
        self.std = std.values if isinstance(std, pd.Series) else std
        self.min_val = min_val.values if isinstance(min_val, pd.Series) else min_val
        self.max_val = max_val.values if isinstance(max_val, pd.Series) else max_val

    def normalize(self, arr):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            return (arr - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            return (arr - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

#         elif self.norm_type == "per_sample_std":
#             grouped = df.groupby(by=df.index)
#             return (df - grouped.transform('mean')) / grouped.transform('std')

#         elif self.norm_type == "per_sample_minmax":
#             grouped = df.groupby(by=df.index)
#             min_vals = grouped.transform('min')
#             return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
