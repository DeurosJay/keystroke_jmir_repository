from timesnet.data_provider.data_factory2 import data_provider
from timesnet.exp.exp_basic import Exp_Basic
from timesnet.utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import joblib
import pandas as pd
from timesnet.data_provider.uea import subsample, interpolate_missing, Normalizer
from timesnet.utils.focal_loss import FocalLoss
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from timesnet.features import PIRA_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict, Counter
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

def clean_nan_inf(X):
            # X shape: (N, T, D)
            N, T, D = X.shape
            reshaped = X.reshape(-1, D)

            # Reemplazar inf por NaN
            reshaped[np.isinf(reshaped)] = np.nan

            # Reemplazar NaNs con la media de cada columna
            if np.isnan(reshaped).any():
                col_means = np.nanmean(reshaped, axis=0)
                inds = np.where(np.isnan(reshaped))
                reshaped[inds] = np.take(col_means, inds[1])

            return reshaped.reshape(N, T, D)

def split_keys_by_labels(window_keys, labels, max_attempts=20):
    key_to_labels = defaultdict(list)
    for key, label in zip(window_keys, labels):
        key_to_labels.setdefault(key, []).append(label)

    key_to_dominant_label = {
        k: Counter(v).most_common(1)[0][0]
        for k, v in key_to_labels.items()
    }

    unique_keys = list(key_to_dominant_label.keys())
    key_labels = [key_to_dominant_label[k] for k in unique_keys]

    print("Distribución de clases en las keys:", Counter(key_labels))
    print("Total keys:", len(unique_keys))

    for attempt in range(max_attempts):
        try:
            if min(Counter(key_labels).values()) < 2:
                print("⚠️ No se puede estratificar en train/temp: clase con < 2 pacientes")
                train_keys, temp_keys = train_test_split(unique_keys, test_size=0.3, random_state=attempt)
            else:
                train_keys, temp_keys = train_test_split(
                    unique_keys, test_size=0.3, stratify=key_labels, random_state=attempt
                )

            temp_labels = [key_to_dominant_label[k] for k in temp_keys]
            if min(Counter(temp_labels).values()) < 2:
                print("⚠️ No se puede estratificar en val/test: clase con < 2 pacientes")
                val_keys, test_keys = train_test_split(temp_keys, test_size=0.5, random_state=attempt)
            else:
                val_keys, test_keys = train_test_split(
                    temp_keys, test_size=0.5, stratify=temp_labels, random_state=attempt
                )

            val_labels = [key_to_dominant_label[k] for k in val_keys]
            test_labels = [key_to_dominant_label[k] for k in test_keys]

            if len(set(val_labels)) < 2:
                print(f"Intento {attempt+1}: val sin ambas clases. Reintentando...")
                continue

            if len(set(test_labels)) < 2:
                print(f"Intento {attempt+1}: test sin ambas clases. Reintentando...")
                continue

            # 🎉 Todo OK
            print("Distribución en train:", Counter([key_to_dominant_label[k] for k in train_keys]))
            print("Distribución en val:", Counter([key_to_dominant_label[k] for k in val_keys]))
            print("Distribución en test:", Counter([key_to_dominant_label[k] for k in test_keys]))
            return train_keys, val_keys, test_keys

        except ValueError:
            continue

    raise RuntimeError("No se pudo generar un split con ambas clases en test después de varios intentos.")

def get_indices_safe(window_keys, keys):
    window_keys = np.array(window_keys)  # asegura que sea array
    return np.where(np.isin(window_keys, keys))[0]

def create_dataset(args, df_raw, features):
    setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_po{args.pira_offset}_ms{args.min_seq_len}_dm{args.d_model}_do{args.dropout}_el{args.e_layers}_df{args.d_ff}_lr{args.learning_rate}_bs{args.batch_size}_sw{args.seq_len}_pw{args.pred_window}_cw{args.class_weight[0]}_{args.class_weight[1]}"
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = os.path.join(path, "div_dataset.txt")
    log = open(log_file, "w")  # o "a" para append
    if args.mode == 'normal':
        scaler = StandardScaler()
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw['Date_CDP'] = pd.to_datetime(df_raw['Date_CDP'])
        df_raw['weekday'] = df_raw['date'].dt.weekday  # 0 (lunes) a 6 (domingo)

        sequences = []
        labels = []
        window_keys = []
        used_keys = set()
        skipped = 0
        skipped_buffer = 0
        for key, group in df_raw.groupby('key'):
            if len(group) < args.seq_len + args.pred_window:
                skipped += 1
                continue
            group = group.sort_values('date').reset_index(drop=True)
            pira_date = group['Date_CDP'].dropna().unique()
            pira_start = pd.to_datetime(pira_date[0]) - pd.Timedelta(days=args.pira_offset) if len(pira_date) > 0 else None

            # [FIX D2] Paso configurable entre ventanas (por defecto 1 = comportamiento
            # anterior). Un stride > 1 reduce el solapamiento y la autocorrelación
            # entre muestras, acercando el nº de ventanas al tamaño muestral efectivo.
            _stride = max(1, int(getattr(args, 'window_stride', 1)))
            for i in range(0, len(group) - args.seq_len - args.pred_window + 1, _stride):
                window = group.iloc[i:i + args.seq_len]
                future = group.iloc[i + args.seq_len:i + args.seq_len + args.pred_window]
                future_start = future['date'].min()
                future_end = future['date'].max()

                # [FIX L3] Se elimina la normalización por ventana con un
                # `normalizer` que nunca se declaraba en esta rama (NameError /
                # global colgante). La estandarización se hace tras el split.
                X_window = window[features].values.astype(np.float32)

                if pira_start:
                    days_to_pira = (pira_start - future_end).days
                    if 0 <= (pira_start - future_start).days <= args.pred_window:
                        label = 1
                    elif days_to_pira < args.buffer_days:
                        skipped_buffer += 1
                        continue
                    else:
                        label = 0
                else:
                    label = 0

                sequences.append(X_window)
                labels.append(label)
                window_keys.append(key)
                used_keys.add(key)

        print(f"Total keys en el CSV: {df_raw['key'].nunique()}")
        print(f"Keys con ventanas suficientes: {len(used_keys)}")
        print(f"Keys saltados por datos insuficientes: {skipped}")
        print(f"Ventanas saltadas por PIRA: {skipped_buffer}")

        # Convertimos a arrays
        X_arr = np.array(sequences)
        y_arr = np.array(labels)
        window_keys = np.array(window_keys)

        # Dimensiones
        N, T, D = X_arr.shape
        reshaped = X_arr.reshape(-1, D)

        # Reemplazar inf por NaN (sin estadísticos; la imputación se hace tras el split)
        reshaped[np.isinf(reshaped)] = np.nan

        # === DIVISIÓN POR PACIENTE (antes de normalizar) ===
        train_keys, val_keys, test_keys = split_keys_by_labels(window_keys, labels)

        train_idx = get_indices_safe(window_keys, train_keys)
        val_idx = get_indices_safe(window_keys, val_keys)
        test_idx = get_indices_safe(window_keys, test_keys)

        # [FIX L1] Imputación de NaN con la media de columnas SOLO de train (antes se
        # usaba la media global sobre train+val+test antes del split -> fuga residual).
        if np.isnan(reshaped).any():
            col_means = np.nanmean(X_arr[train_idx].reshape(-1, D), axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(reshaped))
            reshaped[inds] = np.take(col_means, inds[1])

        # === [FIX L2 / I1] StandardScaler ajustado SOLO sobre train ===
        # Antes: (1) fit_transform global sobre TODAS las ventanas antes del split
        # (fuga de val/test) y (2) un SEGUNDO fit_transform sobre datos YA escalados,
        # cuyo scaler era el que se persistía (incoherente con la transformación real
        # -> entradas sin escalar en inferencia -> salidas fuera de rango).
        # Ahora se ajusta una única vez sobre train y se transforma cada split por
        # separado; ese mismo `scaler` es el que se guarda y se aplica en inferencia.
        scaler = StandardScaler()
        scaler.fit(X_arr[train_idx].reshape(-1, D))

        X_train = scaler.transform(X_arr[train_idx].reshape(-1, D)).reshape(len(train_idx), T, D)
        X_val = scaler.transform(X_arr[val_idx].reshape(-1, D)).reshape(len(val_idx), T, D)
        X_test = scaler.transform(X_arr[test_idx].reshape(-1, D)).reshape(len(test_idx), T, D)
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]
        y_test = y_arr[test_idx]

        # X_all normalizado con el MISMO scaler de train (histórico completo)
        X_all = scaler.transform(X_arr.reshape(-1, D)).reshape(N, T, D)
        y_all = y_arr
        key_all = window_keys

        print("Train keys únicas:", np.unique(train_keys))
        log.write(f"Train keys únicas: {np.unique(train_keys)}\n")
        print("Val keys únicas:", np.unique(val_keys))
        log.write(f"Val keys únicas: {np.unique(val_keys)}\n")
        print("Test keys únicas:", np.unique(test_keys))
        log.write(f"Test keys únicas: {np.unique(test_keys)}\n")
        log.flush()

        return X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all, key_all, scaler

    elif args.mode == 'window':
        scaler = None  # No usamos StandardScaler global para ventanas variables

        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw['Date_CDP'] = pd.to_datetime(df_raw['Date_CDP'])
        df_raw['weekday'] = df_raw['date'].dt.weekday  # 0 (lunes) a 6 (domingo)

        sequences, labels, window_keys = [], [], []
        used_keys = set()
        skipped = 0
        skipped_buffer = 0
        # [FIX L1] Solo se convierte inf -> NaN. NO se imputa ni se calculan
        # estadísticos aquí: media/std e imputación se ajustan DESPUÉS del split,
        # exclusivamente sobre pacientes de train (ver más abajo). Antes,
        # global_mean/global_std se calculaban sobre TODO el dataset (incl. val+test),
        # filtrando información de test al entrenamiento (fuga de datos).
        df_raw[features] = df_raw[features].replace([np.inf, -np.inf], np.nan)

        for key, group in tqdm(df_raw.groupby('key')):
            if len(group) < args.min_seq_len + args.pred_window:
                skipped += 1
                continue

            group = group.sort_values('date').reset_index(drop=True)
            pira_dates = group['Date_CDP'].dropna().unique()
            pira_start = pd.to_datetime(pira_dates[0]) - pd.Timedelta(days=args.pira_offset) if len(pira_dates) > 0 else None

            max_len = min(args.seq_len, len(group) - args.pred_window)

            # [FIX D2] Paso configurable entre ventanas (por defecto 1).
            _stride = max(1, int(getattr(args, 'window_stride', 1)))
            for window_len in range(args.min_seq_len, max_len + 1):
                for i in range(0, len(group) - window_len - args.pred_window + 1, _stride):
                    window = group.iloc[i:i + window_len]
                    future = group.iloc[i + window_len:i + window_len + args.pred_window]
                    future_start = future['date'].min()
                    future_end = future['date'].max()
                    
                    # [FIX L1] Se guarda la ventana SIN normalizar (raw); la
                    # estandarización se aplica tras el split con estadísticos de train.
                    X_window = window[features].values.astype(np.float32)

                    if pira_start:
                        days_to_pira = (pira_start - future_end).days
                        if 0 <= (pira_start - future_start).days <= args.pred_window:
                            label = 1
                        elif days_to_pira < args.buffer_days:
                            skipped_buffer += 1
                            continue
                        else:
                            label = 0
                    else:
                        label = 0

                    # [FIX D1] No se descartan ventanas por NaN/inf: se imputan más
                    # abajo con la media de train (antes se perdían en silencio,
                    # sesgando el dataset sin dejar traza).
                    sequences.append(X_window)
                    labels.append(label)
                    window_keys.append(key)
                    used_keys.add(key)

        print(f"Total keys en el CSV: {df_raw['key'].nunique()}")
        print(f"Keys con ventanas suficientes: {len(used_keys)}")
        print(f"Keys saltados por datos insuficientes: {skipped}")
        print(f"Ventanas saltadas por PIRA: {skipped_buffer}")
        
        print("Total ventanas generadas:", len(sequences))

        seq_arr = np.array(sequences, dtype=object)  # ventanas RAW de longitud variable
        labels_arr = np.array(labels)

        # === DIVISIÓN POR PACIENTE (split a nivel de paciente, igual que antes) ===
        train_keys, val_keys, test_keys = split_keys_by_labels(window_keys, labels)

        train_idx = get_indices_safe(window_keys, train_keys)
        val_idx = get_indices_safe(window_keys, val_keys)
        test_idx = get_indices_safe(window_keys, test_keys)

        # === [FIX L1 / L3 / D1] Estadísticos de normalización SOLO de train ===
        # Se apilan todas las filas (time-steps) de las ventanas de train y se
        # calculan media/desv. típica ignorando NaN. La std nula se sustituye por 1
        # para evitar divisiones que explotaban a inf (causa del descarte silencioso).
        train_rows = np.concatenate(
            [np.asarray(seq_arr[i], dtype=np.float32) for i in train_idx], axis=0
        )
        train_rows = np.where(np.isinf(train_rows), np.nan, train_rows)
        feat_mean = np.nanmean(train_rows, axis=0)
        feat_mean = np.where(np.isnan(feat_mean), 0.0, feat_mean).astype(np.float32)
        feat_std = np.nanstd(train_rows, axis=0)
        feat_std = np.where((feat_std == 0) | np.isnan(feat_std), 1.0, feat_std).astype(np.float32)

        normalizer = Normalizer(norm_type='standardization', mean=feat_mean, std=feat_std)

        def _impute_and_normalize(arr):
            # [FIX D1 / I3] inf -> NaN, imputación con la media de TRAIN y
            # estandarización con el MISMO normalizer que se persiste para inferencia.
            a = np.asarray(arr, dtype=np.float32).copy()
            a = np.where(np.isinf(a), np.nan, a)
            nan_idx = np.where(np.isnan(a))
            if nan_idx[0].size:
                a[nan_idx] = np.take(feat_mean, nan_idx[1])
            return normalizer.normalize(a).astype(np.float32)

        X_train = [_impute_and_normalize(seq_arr[i]) for i in train_idx]
        X_val = [_impute_and_normalize(seq_arr[i]) for i in val_idx]
        X_test = [_impute_and_normalize(seq_arr[i]) for i in test_idx]
        y_train = [labels_arr[i] for i in train_idx]
        y_val = [labels_arr[i] for i in val_idx]
        y_test = [labels_arr[i] for i in test_idx]

        # Todas las secuencias, normalizadas con el normalizer ajustado en train
        X_all = [_impute_and_normalize(s) for s in sequences]
        y_all = labels_arr
        key_all = window_keys

        print("Train keys únicas:", np.unique(train_keys))
        log.write(f"Train keys únicas: {np.unique(train_keys)}\n")
        print("Val keys únicas:", np.unique(val_keys))
        log.write(f"Val keys únicas: {np.unique(val_keys)}\n")
        print("Test keys únicas:", np.unique(test_keys))
        log.write(f"Test keys únicas: {np.unique(test_keys)}\n")
        log.flush()
        
        return X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all, key_all, normalizer
    
class Exp_Classification(Exp_Basic):
    def __init__(self,df_raw=None,args=None,flag='train'):

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        features = list(PIRA_FEATURES)  # [FIX B2] fuente única en timesnet/features.py

                
        if flag == 'train':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.X_all, self.y_all, self.key_all, self.scaler = create_dataset(args, df_raw, features)
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, X=None, y=None):
        data_set, data_loader = data_provider(self.args, flag, X, y)
        return data_set, data_loader

    def _select_optimizer(self):
        # [FIX T4] Optimizador configurable vía args.optimizer (por defecto RAdam,
        # que es lo que se venía usando). Antes estaba fijado en el código pese a
        # poder parametrizarse; ahora la elección es explícita y trazable.
        name = str(getattr(self.args, 'optimizer', 'radam')).lower()
        if name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    # def _select_criterion(self):
    #     criterion = nn.CrossEntropyLoss()
    #     return criterion
    
    def _select_criterion(self, class_weight=[1.0, 5.0], use_focal=False, gamma=2.0):
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(self.device)
        if use_focal:
            return FocalLoss(weight=weight_tensor, gamma=gamma)
        else:
            return nn.CrossEntropyLoss(weight=weight_tensor)
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                #loss = criterion(pred, label.long().squeeze().cpu())
                loss = criterion(pred.to(self.device), label.view(-1).long().to(self.device))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # [FIX B1] dim explicito | (total_samples, num_classes) est. prob. for each class and sample
        probs_class_1 = probs[:, 1].cpu().numpy()  # riesgo de PIRA
        probs_class_0 = probs[:, 0].cpu().numpy()  # no riesgo de PIRA
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy, preds, trues, probs_class_0, probs_class_1

    def train(self, setting, args=None):
        train_data, train_loader = self._get_data('train', self.X_train, self.y_train)
        vali_data, vali_loader = self._get_data('val', self.X_val, self.y_val)
        test_data, test_loader = self._get_data('test', self.X_test, self.y_test)

        path = os.path.join(self.args.checkpoints, setting)
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Crear archivo de log
        log_file = os.path.join(path, "train_log.txt")
        log = open(log_file, "w")  # o "a" para append

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        #criterion = self._select_criterion(args.class_weight)
        # Nota [T4]: la ponderación de clase se aplica UNA sola vez, como `weight`
        # dentro del focal loss (focal ponderado). No hay doble ponderación.
        criterion = self._select_criterion(args.class_weight, use_focal=True)

        # [FIX T4] AMP honesto: se honra args.use_amp SOLO si hay CUDA; en CPU queda
        # como no-op. Antes use_amp=true se ignoraba por completo (config a la deriva).
        use_amp = bool(getattr(self.args, 'use_amp', False)) and (self.device.type == 'cuda')
        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # [FIX T4] Forward/loss bajo autocast cuando AMP está activo.
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    log.write("titers: {0}, epoch: {1} | loss: {2:.7f}\n".format(i + 1, epoch + 1, loss.item()))
                    log.flush()
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    log.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s\n'.format(speed, left_time))
                    log.flush()
                    iter_count = 0
                    time_now = time.time()

                # [FIX T4] Backward/step compatibles con AMP (unscale antes del
                # gradient clipping para preservar el recorte con max_norm=4.0).
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(model_optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                amp_scaler.step(model_optim)
                amp_scaler.update()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            log.write("Epoch: {} cost time: {}\n".format(epoch + 1, time.time() - epoch_time))
            log.flush()
            train_loss = np.average(train_loss)
            # vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            # test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
            vali_loss, val_accuracy, val_preds, val_trues, _, _ = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy, _, _, _, _ = self.vali(test_data, test_loader, criterion)

            # [FIX T2] Criterio de parada sensible al desbalance: F1 macro sobre
            # validación en lugar de accuracy (que premiaba la clase mayoritaria y
            # seleccionaba modelos ciegos a los eventos PIRA, la clase minoritaria).
            val_pred_labels = torch.argmax(val_preds, dim=1).cpu().numpy()
            val_f1 = f1_score(val_trues, val_pred_labels, average='macro', zero_division=0)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1(macro): {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_f1, test_loss, test_accuracy))
            log.write(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1(macro): {5:.3f} Test Loss: {6:.3f} Test Acc: {7:.3f}\n"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_f1, test_loss, test_accuracy))
            log.flush()
            early_stopping(-val_f1, self.model, path, log)
            if early_stopping.early_stop:
                print("Early stopping")
                log.write("Early stopping\n")
                log.flush()
                break

            # [FIX T1] Scheduler de learning rate activo (lradj del config, p.ej.
            # "cosine"). Antes se importaba pero nunca se llamaba: el LR quedaba
            # constante y se ignoraba la configuración.
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        joblib.dump(self.scaler, path + '/' +  f'scaler_{args.model_id}.pkl')
        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test', self.X_test, self.y_test)
        if test:
            print('loading model')
            # [FIX ruta] Cargar desde la MISMA ruta donde train() guarda
            # (self.args.checkpoints), no una ruta './checkpoints/' fija que no
            # coincide si args.checkpoints es distinto -> cargaría un modelo erróneo.
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        log_file = os.path.join(folder_path, "test_log.txt")
        log = open(log_file, "w")  # o "a" para append

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)
        log.write('test shape: {}, {}\n'.format(preds.shape, trues.shape))

        probs = torch.nn.functional.softmax(preds, dim=1)  # [FIX B1] dim explicito | (total_samples, num_classes) est. prob. for each class and sample
        probs_class_1 = probs[:, 1].cpu().numpy()  # riesgo de PIRA
        probs_class_0 = probs[:, 0].cpu().numpy()  # no riesgo de PIRA        
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        df_results = pd.DataFrame({
            'true_label': trues,
            'predicted': predictions,
            'no_risk_score': probs_class_0,
            'risk_score': probs_class_1
        })
        df_results.to_csv('./test_results/' + setting + '/risk_predictions.csv', index=False)
        cm = confusion_matrix(trues, predictions)

        confusion_matrix_file = './test_results/' + setting + '/confusion_matrix.txt'
        if not os.path.exists(os.path.dirname(confusion_matrix_file)):
            os.makedirs(os.path.dirname(confusion_matrix_file))
        
        with open(confusion_matrix_file, 'w') as f:
            f.write('Confusion Matrix for {}\n'.format(setting))
            f.write('--------------------------------\n')
            
            # Imprimir la matriz de confusión en el formato adecuado
            f.write('Predicción\n')
            f.write('            0     1\n')  # Header for the predictions
            f.write('Real\n')
            for i in range(cm.shape[0]):
                f.write(f'{i}  ')
                f.write('  '.join([str(val) for val in cm[i]]) + '\n')
            
            # Aciertos por clase (diagonal principal)
            f.write('\nClass-wise accuracies (Diagonal values):\n')
            for i in range(cm.shape[0]):
                f.write(f'Class {i}: {cm[i, i]} correct predictions\n')

            # Agregar el classification_report
            f.write('\nClassification Report:\n')
            report = classification_report(trues, predictions, target_names=['Class 0', 'Class 1'])
            f.write(report)
        
        print('accuracy:{}'.format(accuracy))
        log.write('accuracy: {}\n'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return