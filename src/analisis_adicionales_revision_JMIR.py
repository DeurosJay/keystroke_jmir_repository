import sys
import os
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parents[1]   # raíz del repositorio
while ROOT_DIR.name != "Modelo predictivo":
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))

from src.utils.EMCare_predictor import EMCarePredictor
# from <MODULO> import normalizer  # TODO: confirmar de dónde viene `normalizer` usado en create_dataset

# === CONFIG ===
checkpoint_name = "classification_timesnet_pira_alcion_repl_20260824_TimesNet_pira_sl90_po30_ms60_dm64_do0.2_el2_df128_lr0.0001_bs1024_sw90_pw120_cw1.0_6.0"
data_dir       = ROOT_DIR / "data"
checkpoint_dir = ROOT_DIR / "timesnet" / "checkpoints" / checkpoint_name
resultados_dir = ROOT_DIR / "src" / "resultados_eval"

model_path = checkpoint_dir / "checkpoint.pth"
scaler_path = checkpoint_dir / "scaler_timesnet_pira_alcion_repl_20260824.pkl"
args_path = checkpoint_dir / "args.json"

predictor = EMCarePredictor(args_path=str(args_path), model_path=str(model_path), scaler_path=str(scaler_path))
model = predictor.model
args = predictor.args
features = predictor.features
scaler = predictor.scaler  # <-- el scaler YA AJUSTADO en entrenamiento, no uno nuevo

VAL_KEYS = {}

# === Cargar merged_all ===
data_dir = Path(r"C:\Users\jlchi\OneDrive\Desktop\EMcare\Modelo predictivo\EMCare\EMCARE\data")
merged_all = pd.read_csv(data_dir / "merged_daily_1.0.csv")  # TODO: confirmar extensión real

print(merged_all.columns.tolist())
print(merged_all.head())
print(merged_all.shape)

print(type(predictor.scaler))
print(getattr(predictor.scaler, "mean", "no tiene mean"))
print(getattr(predictor.scaler, "std", "no tiene std"))
print(getattr(predictor.scaler, "norm_type", "no tiene norm_type"))

# === Construcción de ventanas SOLO para las keys de validación ===
# (misma lógica que dentro de create_dataset, pero sin split ni re-fit de scaler)

normalizer = predictor.scaler  # Normalizer con mean/std reales de entrenamiento

def build_windows_for_val_keys(args, df_raw, features, val_keys, normalizer):
    df_raw = df_raw.copy()
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['Date_CDP'] = pd.to_datetime(df_raw['Date_CDP'])
    df_raw['weekday'] = df_raw['date'].dt.weekday

    sequences, labels, window_keys = [], [], []

    for key, group in df_raw.groupby('key'):
        if key not in val_keys:
            continue
        if len(group) < args.seq_len + args.pred_window:
            continue

        group = group.sort_values('date').reset_index(drop=True)
        pira_date = group['Date_CDP'].dropna().unique()
        pira_start = (pd.to_datetime(pira_date[0]) - pd.Timedelta(days=args.pira_offset)
                      if len(pira_date) > 0 else None)

        for i in range(len(group) - args.seq_len - args.pred_window + 1):
            window = group.iloc[i:i + args.seq_len]
            future = group.iloc[i + args.seq_len:i + args.seq_len + args.pred_window]
            future_start, future_end = future['date'].min(), future['date'].max()

            X_window = window[features].values.astype(np.float32)
            df_window = pd.DataFrame(X_window, columns=features)
            X_window_norm = normalizer.normalize(df_window)  # usa mean/std de entrenamiento
            X_window = X_window_norm.values.astype(np.float32)

            if pira_start:
                days_to_pira = (pira_start - future_end).days
                if 0 <= (pira_start - future_start).days <= args.pred_window:
                    label = 1
                elif days_to_pira < args.buffer_days:
                    continue
                else:
                    label = 0
            else:
                label = 0
            sequences.append(X_window)
            labels.append(label)
            window_keys.append(key)

    X_arr = np.array(sequences)
    y_arr = np.array(labels)
    window_keys = np.array(window_keys)

    N, T, D = X_arr.shape
    reshaped = X_arr.reshape(-1, D)
    reshaped[np.isinf(reshaped)] = np.nan
    if np.isnan(reshaped).any():
        col_means = np.nanmean(reshaped, axis=0)
        inds = np.where(np.isnan(reshaped))
        reshaped[inds] = np.take(col_means, inds[1])
    X_arr = reshaped.reshape(N, T, D)  # ← sin scaler.transform() adicional

    return X_arr, y_arr, window_keys

X_val, y_val, val_window_keys = build_windows_for_val_keys(args, merged_all, features, VAL_KEYS, normalizer)

# Verificación: ¿aparecen las 9 keys esperadas?
found_keys = set(np.unique(val_window_keys))
print("Keys de validación encontradas:", found_keys)
if found_keys != VAL_KEYS:
    print("⚠️ Diferencia respecto a lo esperado:", VAL_KEYS.symmetric_difference(found_keys))

# === Inferencia directa sobre X_val ===
device = next(model.parameters()).device
model.eval()
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
padding_mask = torch.ones(X_val_t.shape[0], X_val_t.shape[1]).to(device)  # TODO: confirmar si hay padding real

with torch.no_grad():
    outputs = model(X_val_t, padding_mask, None, None)
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

val_predictions = probs
y_true = y_val

# === compute_metrics (igual que antes) ===
def compute_metrics(y_true, y_pred_output, class_names=("Class 0", "Class 1")):
    y_pred_output = np.asarray(y_pred_output)
    if y_pred_output.ndim == 2 and y_pred_output.shape[1] > 1:
        risk_score = y_pred_output[:, 1]
        predictions = np.argmax(y_pred_output, axis=1)
    else:
        predictions = y_pred_output.astype(int)
        risk_score = None

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions),
        "classification_report": classification_report(y_true, predictions, target_names=list(class_names)),
    }
    if risk_score is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, risk_score)
        except ValueError:
            pass
    return metrics

val_metrics = compute_metrics(y_true, val_predictions)
print(f"Accuracy: {val_metrics['accuracy']:.4f}")
if "auc" in val_metrics:
    print(f"AUC: {val_metrics['auc']:.4f}")
print("\nMatriz de confusión:\n", val_metrics["confusion_matrix"])
print("\nClassification Report:\n", val_metrics["classification_report"])