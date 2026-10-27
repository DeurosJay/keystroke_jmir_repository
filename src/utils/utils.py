import torch
import numpy as np
import pandas as pd
import json
from argparse import Namespace
from timesnet.exp.exp_classification import Exp_Classification
from timesnet.data_provider.uea import Normalizer
from timesnet.features import PIRA_FEATURES
import joblib

def load_gpu(args):
    """
    Load the appropriate GPU device if available.
    
    Returns:
        torch.device: The device to be used for computations.
    """
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    return args

def load_args(args_file):
    """
    Load arguments from a JSON file.
    
    Args:
        args_file (str): Path to the JSON file containing arguments.
        
    Returns:
        Namespace: Arguments loaded from the JSON file.
    """
    with open(args_file, "r") as f:
        args_dict = json.load(f)
    return Namespace(**args_dict)

def load_model_scaler(model_file, scaler_file, args):
    """
    Load the model state from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        args (Namespace): Arguments containing the path to the checkpoint.
        
    Returns:
        torch.nn.Module: The model with loaded state.
    """
    # Cargar modelo y scaler
    exp = Exp_Classification(args=args, flag=args.flag)
    model = exp._build_model()
    model.load_state_dict(torch.load(model_file, map_location=args.device))
    model.to(args.device)
    model.eval()
    scaler = joblib.load(scaler_file)  # Usa tu ruta real

    return model, scaler

def clean_nan_inf(X):
            # X shape: (N, T, D)
            N, T, D = X.shape
            reshaped = X.reshape(-1, D)

            # Reemplazar inf por NaN
            reshaped[np.isinf(reshaped)] = np.nan

            # Reemplazar NaNs con la media de cada columna
            if np.isnan(reshaped).any():
                print("NaNs detected. Replacing with column means...")
                col_means = np.nanmean(reshaped, axis=0)
                inds = np.where(np.isnan(reshaped))
                reshaped[inds] = np.take(col_means, inds[1])

            return reshaped.reshape(N, T, D)

def predict_model(model, scaler, args, df, features):
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df = df.sort_values('date')

    # Extraer la última ventana de tamaño `args.seq_len`
    if len(df) < args.seq_len:
        raise ValueError("No hay suficientes datos en el CSV (necesita al menos {} filas)".format(args.seq_len))
    
    window = df.iloc[-args.seq_len:]
    X = window[features].values.astype(np.float32)
    X = X.reshape(1, args.seq_len, -1)

    # [FIX I1/I2/I3] Escalado idéntico al de entrenamiento, seleccionando el método
    # por el TIPO del scaler persistido (sin `except` que enmascare el fallo) e
    # imputando los NaN/inf con la media almacenada en el propio scaler (train),
    # no con la media de la ventana.
    X_scaled = _scale_like_training(X, scaler)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(args.device)
    padding_mask = torch.ones((1, X_tensor.shape[1]), dtype=torch.float32).to(args.device)

    with torch.no_grad():
        outputs = model(X_tensor, padding_mask, None, None)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probs_class_1 = probs[:, 1].cpu().numpy()  # riesgo de PIRA
        probs_class_0 = probs[:, 0].cpu().numpy()  # no riesgo de PIRA  
        predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class, probs_class_0, probs_class_1

def _scaler_mean(scaler):
    """Media (por feature) almacenada en el scaler persistido, sea Normalizer o
    StandardScaler de sklearn. Se usa para imputar NaN igual que en entrenamiento."""
    if hasattr(scaler, "mean") and scaler.mean is not None:      # Normalizer (modo window)
        return np.asarray(scaler.mean, dtype=np.float32)
    if hasattr(scaler, "mean_"):                                  # StandardScaler (modo normal)
        return np.asarray(scaler.mean_, dtype=np.float32)
    return None


def _scale_like_training(X, scaler):
    """
    [FIX I1/I3] Aplica EXACTAMENTE la transformación de entrenamiento:
      - imputa inf->NaN->media de train (la del scaler), no la de la ventana;
      - estandariza con Normalizer.normalize (modo window) o
        StandardScaler.transform (modo normal), elegido por el tipo del scaler.
    X: np.ndarray (N, T, D)  ->  np.ndarray (N, T, D)
    """
    N, T, D = X.shape
    X_2d = X.reshape(-1, D).astype(np.float32).copy()
    X_2d = np.where(np.isinf(X_2d), np.nan, X_2d)

    mean = _scaler_mean(scaler)
    nan_idx = np.where(np.isnan(X_2d))
    if nan_idx[0].size:
        if mean is not None:
            X_2d[nan_idx] = np.take(mean, nan_idx[1])
        else:
            X_2d[nan_idx] = 0.0

    if hasattr(scaler, "normalize"):        # Normalizer (modo window)
        X_scaled_2d = scaler.normalize(X_2d)
    elif hasattr(scaler, "transform"):      # StandardScaler de sklearn (modo normal)
        X_scaled_2d = scaler.transform(X_2d)
    else:
        raise TypeError(f"Scaler no reconocido para inferencia: {type(scaler)}")

    return np.asarray(X_scaled_2d, dtype=np.float32).reshape(N, T, D)


def _scale_with_normalizer(X, normalizer):
    """Compat.: se mantiene el nombre antiguo; delega en la ruta corregida."""
    return _scale_like_training(X, normalizer)

def predict_ultima_ventana(model, normalizer, args, df, features):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df = df.sort_values('date')

    if len(df) < args.seq_len:
        raise ValueError(f"No hay suficientes datos (necesita ≥ {args.seq_len} filas)")

    window = df.iloc[-args.seq_len:]
    X = window[features].values.astype(np.float32).reshape(1, args.seq_len, -1)

    # [FIX I3-bis] Sin `clean_nan_inf` (que imputaba con la media de la VENTANA):
    # _scale_like_training imputa con la media de TRAIN almacenada en el scaler,
    # idéntico a entrenamiento y a predict_model.
    X_scaled = _scale_with_normalizer(X, normalizer)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=args.device)
    padding_mask = torch.ones((1, X_tensor.shape[1]), dtype=torch.float32, device=args.device)

    with torch.no_grad():
        outputs = model(X_tensor, padding_mask, None, None)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        probs_np = probs.detach().cpu().numpy()[0]
        pred = int(np.argmax(probs_np))
        prob_0, prob_1 = float(probs_np[0]), float(probs_np[1])

    return pred, prob_0, prob_1

def predict_over_time(model, normalizer, args, df, features):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df = df.sort_values('date').reset_index(drop=True)

    preds, probs0, probs1, dates = [], [], [], []

    if len(df) < args.seq_len:
        raise ValueError(f"No hay suficientes datos (necesita ≥ {args.seq_len} filas)")

    for i in range(args.seq_len, len(df) + 1):
        window = df.iloc[i - args.seq_len:i]
        X = window[features].values.astype(np.float32).reshape(1, args.seq_len, -1)

        # [FIX I3-bis] imputación coherente con train vía _scale_like_training (sin clean_nan_inf).
        X_scaled = _scale_with_normalizer(X, normalizer)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=args.device)
        padding_mask = torch.ones((1, X_tensor.shape[1]), dtype=torch.float32, device=args.device)

        with torch.no_grad():
            outputs = model(X_tensor, padding_mask, None, None)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_np = probs.detach().cpu().numpy()[0]
            pred = int(np.argmax(probs_np))

        print(f"Predicción: {pred}, Probabilidad clase 0: {probs_np[0]}, Probabilidad clase 1: {probs_np[1]}")

        preds.append(pred)
        probs0.append(float(probs_np[0]))
        probs1.append(float(probs_np[1]))
        dates.append(window.iloc[-1]['date'])

    results = pd.DataFrame({
        'date': dates,
        'pred': preds,
        'prob_0': probs0,
        'prob_1': probs1
    })
    return results

def predict_model_window(model, args, df, scaler, seq_len=None):
    """
    Predice con una secuencia temporal de longitud variable usando normalización por ventana.
    """

    # === Preprocesado ===
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df = df.sort_values('date')

    # [FIX I4] La cabeza de clasificación es una proyección de tamaño FIJO
    # (d_model * args.seq_len). El tensor de entrada debe tener SIEMPRE longitud
    # args.seq_len: se usan como mucho args.seq_len pasos reales y se rellena/enmascara
    # igual que el collate_fn de entrenamiento (datos reales al inicio, ceros después,
    # máscara 1 en posiciones reales). Antes, una longitud distinta de 90 rompía por
    # mismatch de shape.
    feat_cols = list(PIRA_FEATURES)  # [FIX B2] fuente única en timesnet/features.py

    max_len = int(args.seq_len)
    if seq_len is not None:
        max_len = min(max_len, int(seq_len))
    if len(df) < 1:
        raise ValueError("DataFrame de inferencia vacío")

    window = df.iloc[-max_len:]
    real_len = len(window)
    if real_len < int(getattr(args, 'min_seq_len', 0)):
        print(f"[aviso] ventana de {real_len} pasos < min_seq_len "
              f"({getattr(args, 'min_seq_len', 0)}): fuera de la distribución de entrenamiento")

    X_raw = window[feat_cols].values.astype(np.float32).reshape(1, real_len, len(feat_cols))
    # Imputación + estandarización idénticas a entrenamiento (mismo scaler de train)
    X_scaled = _scale_like_training(X_raw, scaler)

    D = X_scaled.shape[-1]
    X = np.zeros((1, int(args.seq_len), D), dtype=np.float32)
    X[0, :real_len, :] = X_scaled[0]
    padding_mask = torch.zeros((1, int(args.seq_len)), dtype=torch.float32)
    padding_mask[0, :real_len] = 1.0
    padding_mask = padding_mask.to(args.device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(args.device)
        outputs = model(X_tensor, padding_mask, None, None)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        probs_class_0 = probs[:, 0].cpu().numpy()
        probs_class_1 = probs[:, 1].cpu().numpy()

    return pred_class, probs_class_0, probs_class_1


def clean_demografico(df):
    """
    Clean the demographic data DataFrame by removing unnecessary columns and renaming others.
    
    Parameters:
    df (DataFrame): The input DataFrame containing demographic data.
    
    Returns:
    DataFrame: A cleaned DataFrame with only relevant columns.
    """
    df_clean = df.copy()  

    # 1. Cambiar el tipo de datos de las columnas
    df_clean['Paciente'] = df_clean['Paciente'].astype('Int64')
    df_clean['Genero'] = df_clean['Genero'].astype('Int64')
    df_clean['Edad'] = df_clean['Edad'].astype('Float64')
    df_clean['n'] = df_clean['n'].astype('Int64')

    # PIRA_SN: si PIRA_SN == 0, la fecha debe estar vacía
    # Opcion 1: los dejos como estan

    # Opcion 2: los convertimos a 0
    #df_clean['PIRA_SN'] = df_clean['PIRA_SN'].fillna(0).astype('Int64') #### PUEDE QUE lOS NAN SI SEAN IMPOTANTES Y NO PODAMOS PONER 0

    # Opcion 3: los convertimos a 1
    # Marcar PIRA_SN como 0 si también está todo lo demás vacío
    mask_no_pira_info = (
        df_clean['PIRA_SN'].isna() &
        (df_clean['PIRA_events'].fillna(0) == 0) &
        df_clean['Date_CDP'].isna()
    )
    
    df_clean.loc[mask_no_pira_info, 'PIRA_SN'] = 0
    df_clean['PIRA_SN'] = df_clean['PIRA_SN'].astype('Int64')

    df_clean.loc[df_clean['PIRA_SN'] == 0, 'Date_CDP'] = pd.NaT

    # CDP_SN, SI PIRA_SN == 1, CDP_SN debe ser 1, pero si PIRA_SN == 0, CDP_SN puede ser 0 o 1
    df_clean['CDP_SN'] = df_clean['CDP_SN'].fillna(0).astype('Int64')
    df_clean.loc[df_clean['PIRA_SN'] == 1, 'CDP_SN'] = 1

    # date_CDP
    df_clean['Date_CDP'] = pd.to_datetime(df_clean['Date_CDP'], errors='coerce')

    # PIRA_events
    df_clean['PIRA_events'] = df_clean['PIRA_events'].fillna(0).astype('Int64')

    # 3. Codificación de variables categóricas
    df_encoded = pd.get_dummies(df_clean, columns=['FAME_n', 'Tipo_EM'])

    for col in ['FAME_n_High', 'FAME_n_Low', 'FAME_n_No',
                'Tipo_EM_EMPP', 'Tipo_EM_EMRR', 'Tipo_EM_EMSP']:
        df_encoded[col] = df_encoded[col].astype(int)

    # 4. Eliminar pacientes sin duración ('n') o sin ID
    df_encoded = df_encoded.dropna(subset=['n', 'key'])

    # 5. Reset index
    df_encoded = df_encoded.reset_index(drop=True)
    return df_encoded

def merge_dataframes(df, df_demografico):
    """
    Merge two DataFrames on the 'key' column.
    
    Args:
        df (DataFrame): The first DataFrame to merge.
        df_demografico (DataFrame): The second DataFrame to merge.
        
    Returns:
    DataFrame: The merged DataFrame.
    """
    merged_daily = pd.merge(df, df_demografico, on='key', how='left')
    merged_daily['date'] = pd.to_datetime(merged_daily['date'])
    merged_daily['Date_CDP'] = pd.to_datetime(merged_daily['Date_CDP'])
    merged_daily['weekday'] = merged_daily['date'].dt.weekday  # 0 (lunes) a 6 (domingo)
    return merged_daily

def pad_sequence(df, seq_len, features, method="zeros", position="pre"):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    current_len = len(df)
    if current_len >= seq_len:
        return df.iloc[-seq_len:].copy()

    padding_len = seq_len - current_len

    # Generar fechas artificiales
    if position == "pre":
        pad_dates = pd.date_range(end=df['date'].min() - pd.Timedelta(days=1), periods=padding_len)
    else:
        pad_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=padding_len)

    # Crear padding vacío con mismas columnas
    pad_df = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        if col in features:
            if method == "zeros":
                pad_df[col] = [0.0] * padding_len
            elif method == "mean":
                pad_df[col] = [df[col].mean()] * padding_len
            else:
                raise ValueError("Método de padding no reconocido")
        elif col == 'date':
            pad_df[col] = pad_dates
        elif col == 'weekday':
            pad_df[col] = pd.to_datetime(pad_df['date']).dt.weekday
        elif df[col].dtype in [np.int64, np.float64]:
            pad_df[col] = [df[col].mode()[0]] * padding_len
        else:
            pad_df[col] = [df[col].iloc[0]] * padding_len

    if position == "pre":
        df_padded = pd.concat([pad_df, df], ignore_index=True)
    elif position == "post":
        df_padded = pd.concat([df, pad_df], ignore_index=True)
    else:
        raise ValueError("position debe ser 'pre' o 'post'")

    return df_padded
