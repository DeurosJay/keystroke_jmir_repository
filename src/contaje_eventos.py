import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_auc_score
)

from timesnet.data_provider.uea import Normalizer

# === ROOT_DIR ===
ROOT_DIR = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT_DIR)
sys.path.append(str(ROOT_DIR))

from src.utils.EMCare_predictor import EMCarePredictor

# ============================================================
# === 1) CONFIG / CARGA DEL MODELO ===
# ============================================================
checkpoint_name = "classification_timesnet_pira_alcion_repl_20260824_TimesNet_pira_sl90_po30_ms60_dm64_do0.2_el2_df128_lr0.0001_bs1024_sw90_pw120_cw1.0_6.0"
data_dir       = ROOT_DIR / "data"
checkpoint_dir = ROOT_DIR / "timesnet" / "checkpoints" / checkpoint_name
resultados_dir = ROOT_DIR / "src" / "resultados_eval"

model_path = checkpoint_dir / "checkpoint.pth"
scaler_path = checkpoint_dir / "scaler_timesnet_pira_alcion_repl_20260824.pkl"
args_path = checkpoint_dir / "args.json"

predictor = EMCarePredictor(
    args_path=str(args_path),
    model_path=str(model_path),
    scaler_path=str(scaler_path),
)
model = predictor.model
args = predictor.args
features = predictor.features

# El objeto guardado en scaler_path es en realidad un Normalizer
# (con mean/std ya calculados en entrenamiento), no un StandardScaler.
normalizer = predictor.scaler

# ============================================================
# === 2) CARGA DEL DATASET (merged_all, agregado diario) ===
# ============================================================
# TODO: ajustar ruta/nombre exacto del archivo si difiere en tu máquina
data_dir = ROOT_DIR / "data"
merged_all = pd.read_csv(data_dir / "merged_daily_1.0.csv")  # TODO: nombre real del archivo agregado diario

print("Columnas de merged_all:", merged_all.columns.tolist())
print("Shape de merged_all:", merged_all.shape)

# ============================================================
# === 3) KEYS DE TRAIN / VAL / TEST (idénticas al entrenamiento original) ===
# ============================================================
TRAIN_KEYS = {

}

VAL_KEYS = {

}

TEST_KEYS = {

}


# ============================================================
# === 4) CONSTRUCCIÓN DE VENTANAS DE LONGITUD VARIABLE ===
#     (replica el bloque args.mode == 'window' de create_dataset:
#      min_seq_len..seq_len, no una sola ventana fija por posición)
# ============================================================
print("args.mode:", getattr(args, "mode", "NO DEFINIDO"))
print("args.min_seq_len:", getattr(args, "min_seq_len", "NO DEFINIDO"))
print("args.seq_len:", args.seq_len)


def build_variable_windows_for_keys(args, df_raw, features, keys_subset, normalizer):
    """
    Replica el bloque `elif args.mode == 'window':` de create_dataset:
    para cada key, genera TODAS las ventanas de longitud variable entre
    args.min_seq_len y min(args.seq_len, len(group) - args.pred_window),
    deslizando la posición de inicio. Se descartan (no se imputan) las
    ventanas que tras normalizar contienen NaN/inf.
    """
    df_raw = df_raw.copy()
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_raw['Date_CDP'] = pd.to_datetime(df_raw['Date_CDP'])
    df_raw['weekday'] = df_raw['date'].dt.weekday

    # Mismo preprocesado GLOBAL que en entrenamiento (antes de generar ventanas)
    df_raw[features] = df_raw[features].replace([np.inf, -np.inf], np.nan)
    df_raw[features] = df_raw[features].fillna(df_raw[features].mean())

    sequences, labels, window_keys = [], [], []
    skipped, skipped_buffer = 0, 0

    for key, group in df_raw.groupby('key'):
        if key not in keys_subset:
            continue
        if len(group) < args.min_seq_len + args.pred_window:
            skipped += 1
            continue

        group = group.sort_values('date').reset_index(drop=True)
        pira_dates = group['Date_CDP'].dropna().unique()
        pira_start = (pd.to_datetime(pira_dates[0]) - pd.Timedelta(days=args.pira_offset)
                      if len(pira_dates) > 0 else None)

        max_len = min(args.seq_len, len(group) - args.pred_window)

        for window_len in range(args.min_seq_len, max_len + 1):
            for i in range(len(group) - window_len - args.pred_window + 1):
                window = group.iloc[i:i + window_len]
                future = group.iloc[i + window_len:i + window_len + args.pred_window]
                future_start, future_end = future['date'].min(), future['date'].max()

                X_window = window[features].values.astype(np.float32)
                df_window = pd.DataFrame(X_window, columns=features)
                X_window_norm = normalizer.normalize(df_window)
                X_window = X_window_norm.values.astype(np.float32)

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

                if np.isnan(X_window).any() or np.isinf(X_window).any():
                    continue

                sequences.append(X_window)  # longitud variable (window_len, D)
                labels.append(label)
                window_keys.append(key)

    print(f"Keys saltadas por datos insuficientes: {skipped}")
    print(f"Ventanas saltadas por buffer PIRA: {skipped_buffer}")
    print(f"Total ventanas generadas: {len(sequences)}")

    return sequences, np.array(labels), np.array(window_keys)


def pad_windows_to_fixed_length(sequences, seq_len, position="post"):
    """
    Rellena con ceros cada secuencia de longitud variable hasta seq_len,
    y devuelve también la padding_mask correspondiente (True = dato real).

    TODO: confirmar contra el collate_fn / DataLoader real del entrenamiento
    si el padding se aplicaba al principio ("pre") o al final ("post") de
    la secuencia -- aquí se asume "post" (relleno al final) por defecto,
    igual que el valor por defecto de pad_sequence() en utils.py.
    """
    N = len(sequences)
    D = sequences[0].shape[1]
    X_padded = np.zeros((N, seq_len, D), dtype=np.float32)
    padding_mask = np.zeros((N, seq_len), dtype=bool)

    for idx, seq in enumerate(sequences):
        L = seq.shape[0]
        if position == "post":
            X_padded[idx, :L, :] = seq
            padding_mask[idx, :L] = True
        elif position == "pre":
            X_padded[idx, seq_len - L:, :] = seq
            padding_mask[idx, seq_len - L:] = True
        else:
            raise ValueError("position debe ser 'pre' o 'post'")

    return X_padded, padding_mask


train_sequences, y_train, train_window_keys = build_variable_windows_for_keys(
    args, merged_all, features, TRAIN_KEYS, normalizer)
val_sequences, y_val, val_window_keys = build_variable_windows_for_keys(
    args, merged_all, features, VAL_KEYS, normalizer)
test_sequences, y_test, test_window_keys = build_variable_windows_for_keys(
    args, merged_all, features, TEST_KEYS, normalizer)

X_train, train_padding_mask = pad_windows_to_fixed_length(train_sequences, args.seq_len)
X_val, val_padding_mask = pad_windows_to_fixed_length(val_sequences, args.seq_len)
X_test, test_padding_mask = pad_windows_to_fixed_length(test_sequences, args.seq_len)

# === Verificación de que se han encontrado todas las keys esperadas ===
for nombre, esperado, encontrado in [
    ("Train", TRAIN_KEYS, set(np.unique(train_window_keys))),
    ("Val", VAL_KEYS, set(np.unique(val_window_keys))),
    ("Test", TEST_KEYS, set(np.unique(test_window_keys))),
]:
    faltantes = esperado - encontrado
    if faltantes:
        print(f"⚠️ {nombre}: no se generaron ventanas para estas keys: {faltantes}")
    else:
        print(f"✅ {nombre}: todas las keys esperadas generaron ventanas.")

# === Conteos PIRA / No PIRA (a nivel de ventana) ===
train_pira = int(np.sum(y_train == 1))
train_no_pira = int(np.sum(y_train == 0))
val_pira = int(np.sum(y_val == 1))
val_no_pira = int(np.sum(y_val == 0))
test_pira = int(np.sum(y_test == 1))
test_no_pira = int(np.sum(y_test == 0))

print(f"\nTrain - PIRA: {train_pira}, No PIRA: {train_no_pira}, Total: {len(y_train)}")
print(f"Val   - PIRA: {val_pira}, No PIRA: {val_no_pira}, Total: {len(y_val)}")
print(f"Test  - PIRA: {test_pira}, No PIRA: {test_no_pira}, Total: {len(y_test)}")


# ============================================================
# === 5) INFERENCIA DEL MODELO ===
# ============================================================
def predict_with_model(model, X_arr, padding_mask_arr, args, batch_size=1024):
    """
    Inferencia directa sobre un array (N, seq_len, D) ya normalizado y
    paddeado, usando la padding_mask real (True = dato real, False = relleno)
    para que el modelo ignore las posiciones rellenadas.
    Se procesa en lotes para no agotar memoria de GPU con datasets grandes.
    """
    device = next(model.parameters()).device
    model.eval()

    all_probs = []
    N = X_arr.shape[0]
    for start in range(0, N, batch_size):
        end = start + batch_size
        X_batch = torch.tensor(X_arr[start:end], dtype=torch.float32).to(device)
        mask_batch = torch.tensor(padding_mask_arr[start:end], dtype=torch.bool).to(device)

        with torch.no_grad():
            outputs = model(X_batch, mask_batch, None, None)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


train_predictions = predict_with_model(model, X_train, train_padding_mask, args)
val_predictions = predict_with_model(model, X_val, val_padding_mask, args)
test_predictions = predict_with_model(model, X_test, test_padding_mask, args)


# ============================================================
# === 6) MÉTRICAS ===
# ============================================================
def compute_metrics(y_true, y_pred_output, class_names=("Class 0", "Class 1")):
    """
    y_pred_output: array de probabilidades (n_samples, n_clases), salida de softmax.
    """
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
        "classification_report": classification_report(
            y_true, predictions, target_names=list(class_names)
        ),
    }

    if risk_score is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, risk_score)
        except ValueError:
            pass  # p.ej. si solo hay una clase presente en y_true

    return metrics


for nombre, y_true, y_pred in [
    ("TRAIN", y_train, train_predictions),
    ("VAL", y_val, val_predictions),
    ("TEST", y_test, test_predictions),
]:
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n===== {nombre} =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "auc" in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    print("Matriz de confusión:\n", metrics["confusion_matrix"])
    print("Classification Report:\n", metrics["classification_report"])


# ============================================================
# === 7) GUARDAR RESULTADOS PARA ANÁLISIS POSTERIOR (bootstrap, etc.) ===
# ============================================================
# Se guardan y_true, las probabilidades predichas y las keys de ventana de
# cada conjunto, para poder ejecutar análisis adicionales (como el
# bootstrapping por paciente) sin tener que recargar el modelo ni
# regenerar las ventanas cada vez.
resultados_dir = ROOT_DIR / "emcare-ml-fix-git-trainDavid" / "emcare-ml-fix-git" / "src" / "resultados_eval"
resultados_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    resultados_dir / "resultados_eval.npz",
    y_train=y_train, train_predictions=train_predictions, train_window_keys=train_window_keys,
    y_val=y_val, val_predictions=val_predictions, val_window_keys=val_window_keys,
    y_test=y_test, test_predictions=test_predictions, test_window_keys=test_window_keys,
)
print(f"\nResultados guardados en: {resultados_dir / 'resultados_eval.npz'}")