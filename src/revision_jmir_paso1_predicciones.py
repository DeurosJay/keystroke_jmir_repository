"""
REVISIÓN JMIR (2.ª ronda) — PASO 1: ventanas con metadatos + probabilidades del modelo YA ENTRENADO.
NO reentrena nada.

Colócalo en  F:\\emcare-ml-fix-git-trainDavid\\emcare-ml-fix-git\\src  (junto a contaje_eventos.py)
y ejecútalo desde ahí. Las rutas se calculan solas a partir de esa carpeta.

DOS MODOS (MODO, abajo):
  "npz"    (por defecto) REUTILIZA las probabilidades que ya calculó contaje_eventos.py
           (resultados_eval.npz). No carga el modelo ni usa GPU: regenera las ventanas con la
           MISMA lógica de contaje_eventos.py solo para recuperar sus metadatos (fechas, longitud,
           días hasta el evento) y comprueba, ventana a ventana, que paciente y etiqueta coinciden
           con el .npz. Así las cifras son exactamente las mismas que ya teníais.
  "modelo" Vuelve a pasar el modelo (por si el .npz no cuadra o no existe).

Salidas (en src/resultados_revision_jmir/):
  - ventanas_predicciones.csv  -> una fila por ventana (train/val/test)
  - pacientes.csv              -> una fila por paciente (los 99 del CSV, incluidos los excluidos de fase 2)

Después: revision_jmir_paso2_analisis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# === CONFIG ===
# ============================================================
MODO = "npz"                                   # "npz" o "modelo"

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[1]
# === CONFIG ===
checkpoint_name = "classification_timesnet_pira_alcion_repl_20260824_TimesNet_pira_sl90_po30_ms60_dm64_do0.2_el2_df128_lr0.0001_bs1024_sw90_pw120_cw1.0_6.0"
data_dir       = ROOT_DIR / "data"
checkpoint_dir = ROOT_DIR / "timesnet" / "checkpoints" / checkpoint_name
resultados_dir = ROOT_DIR / "src" / "resultados_eval"

model_path = checkpoint_dir / "checkpoint.pth"
scaler_path = checkpoint_dir / "scaler_timesnet_pira_alcion_repl_20260824.pkl"
args_path = checkpoint_dir / "args.json"

# contaje_eventos.py guardó el .npz en una ruta "anidada"; se busca en ambos sitios.
NPZ_CANDIDATES = [
    ROOT_DIR / "emcare-ml-fix-git-trainDavid" / "emcare-ml-fix-git" / "src" / "resultados_eval" / "resultados_eval.npz",
    SRC_DIR / "resultados_eval" / "resultados_eval.npz",
]

# Imputación de NaN/inf (solo modo "modelo"; en modo "npz" se usa la de contaje_eventos.py = media global)
#   "train_mean"  -> media de TRAIN (= normalizer.mean), como exp_classification.py corregido [FIX L1]
#   "global_mean" -> media de todo el CSV, como contaje_eventos.py
IMPUTACION = "global_mean"
BATCH_SIZE = 1024

TRAIN_KEYS = {
    }
VAL_KEYS = {}
TEST_KEYS = {}

FEATURES = [
    'mean_added_count', 'median_added_count', 'max_added_count', 'min_added_count',
    'mean_removed_count', 'median_removed_count', 'max_removed_count', 'min_removed_count',
    'mean_index', 'median_index', 'max_index', 'min_index',
    'Edad', 'Genero', 'FAME_n_High', 'FAME_n_Low', 'FAME_n_No',
    'Tipo_EM_EMPP', 'Tipo_EM_EMRR', 'Tipo_EM_EMSP']
STATIC = ['Edad', 'Genero', 'FAME_n_High', 'FAME_n_Low', 'FAME_n_No',
          'Tipo_EM_EMPP', 'Tipo_EM_EMRR', 'Tipo_EM_EMSP']
OPTIONAL_STATIC = ['PST', 'PST_basal', 'EDSS', 'EDSS_basal']   # se copian si existen en el CSV
COLS = ['key', 'conjunto', 'longitud', 'idx_inicio', 'fecha_inicio_input', 'fecha_fin_input',
        'fecha_inicio_horizonte', 'fecha_fin_horizonte', 'label', 'dias_fin_input_a_cdp']


def set_of(key):
    return 'train' if key in TRAIN_KEYS else 'val' if key in VAL_KEYS else 'test' if key in TEST_KEYS else 'fuera_fase2'


def load_args():
    with open(CHECKPOINT_DIR / "args.json") as f:
        a = json.load(f)
    return {k: int(a[k]) for k in ['seq_len', 'min_seq_len', 'pred_window', 'pira_offset', 'buffer_days']}


def load_model():
    sys.path.insert(0, str(ROOT_DIR))
    from src.utils.EMCare_predictor import EMCarePredictor
    p = EMCarePredictor(args_path=str(CHECKPOINT_DIR / "args.json"),
                        model_path=str(CHECKPOINT_DIR / "checkpoint.pth"),
                        scaler_path=str(CHECKPOINT_DIR / SCALER_FILE))
    return p.model, p.scaler


def predict_probs(model, seqs, seq_len):
    """Padding al final + padding_mask (1 = dato real), igual que collate_fn(mode='window')."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    out = []
    for s in range(0, len(seqs), BATCH_SIZE):
        chunk = seqs[s:s + BATCH_SIZE]
        X = np.zeros((len(chunk), seq_len, chunk[0].shape[1]), dtype=np.float32)
        M = np.zeros((len(chunk), seq_len), dtype=np.float32)
        for j, w in enumerate(chunk):
            L = min(w.shape[0], seq_len)
            X[j, :L] = w[:L]
            M[j, :L] = 1.0
        with torch.no_grad():
            o = model(torch.tensor(X).to(device), torch.tensor(M).to(device), None, None)
            out.append(torch.softmax(o, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def build(df_raw, a, normalizer=None, imput="global_mean"):
    """Genera ventanas con la lógica de create_dataset/contaje_eventos (mode='window').
    Devuelve (lista de metadatos, lista de arrays normalizados o None, tabla de pacientes)."""
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['Date_CDP'] = pd.to_datetime(df['Date_CDP'])
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if imput == "global_mean":
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].mean())
    if normalizer is not None:
        mean = np.asarray(normalizer.mean, dtype=np.float32)
        std = np.asarray(normalizer.std, dtype=np.float32)

    sl, ml, pw, off, buf = a['seq_len'], a['min_seq_len'], a['pred_window'], a['pira_offset'], a['buffer_days']
    meta, seqs, pats = [], [], []
    for key, g in df.groupby('key'):                      # mismo orden que contaje_eventos.py
        g = g.sort_values('date').reset_index(drop=True)
        conj = set_of(key)
        cdp = g['Date_CDP'].dropna().unique()
        cdp = pd.Timestamp(cdp[0]) if len(cdp) else pd.NaT
        n = len(g)
        span = (g['date'].iloc[-1] - g['date'].iloc[0]).days + 1
        pr = dict(key=key, conjunto=conj, pira=int(pd.notna(cdp)), fecha_cdp=cdp, n_dias_con_datos=n,
                  primer_dia=g['date'].iloc[0], ultimo_dia=g['date'].iloc[-1], dias_calendario=span,
                  pct_dias_sin_datos=round(100 * (1 - n / span), 1),
                  ts_basal_30d=g['mean_added_count'].iloc[:30].median(),
                  rk_basal_30d=g['mean_removed_count'].iloc[:30].median(),
                  index_basal_30d=g['mean_index'].iloc[:30].median(),
                  fame_cambia_en_el_tiempo=int(g[['FAME_n_High', 'FAME_n_Low', 'FAME_n_No']].nunique().max() > 1),
                  **{c: g[c].iloc[0] for c in STATIC},
                  **{c: g[c].iloc[0] for c in OPTIONAL_STATIC if c in g.columns})
        if n < ml + pw:
            pr.update(incluido_fase2=0, motivo_exclusion=f"<{ml + pw} días con datos", n_ventanas=0,
                      n_ventanas_pos=0, n_saltadas_buffer=0, n_ventanas_esperadas_sin_buffer=0)
            pats.append(pr)
            continue
        if conj == 'fuera_fase2':
            print(f"⚠️  {key}: tiene datos suficientes para la fase 2 pero NO está en train/val/test")
            conj = pr['conjunto'] = 'no_asignado'

        X = g[FEATURES].values.astype(np.float32)
        if normalizer is not None:
            if imput == "train_mean":
                r, c = np.where(np.isnan(X)); X[r, c] = mean[c]
            Xn = (X - mean) / (std + np.finfo(float).eps)
        else:
            Xn = X                                   # solo para comprobar NaN (modo npz)
        bad = np.isnan(Xn).any(axis=1) | np.isinf(Xn).any(axis=1)
        dates = g['date'].values
        ps = cdp - pd.Timedelta(days=off) if pd.notna(cdp) else None
        cnt = skipped = pos = 0
        maxL = min(sl, n - pw)
        for L in range(ml, maxL + 1):
            for i in range(0, n - L - pw + 1):
                fs, fe = pd.Timestamp(dates[i + L]), pd.Timestamp(dates[i + L + pw - 1])
                if ps is not None:
                    if 0 <= (ps - fs).days <= pw:
                        lab = 1
                    elif (ps - fe).days < buf:
                        skipped += 1
                        continue
                    else:
                        lab = 0
                else:
                    lab = 0
                if bad[i:i + L].any():               # contaje_eventos.py descarta ventanas con NaN/inf
                    continue
                end = pd.Timestamp(dates[i + L - 1])
                meta.append((key, conj, L, i, pd.Timestamp(dates[i]), end, fs, fe, lab,
                             (cdp - end).days if pd.notna(cdp) else np.nan))
                if normalizer is not None:
                    seqs.append(Xn[i:i + L])
                cnt += 1; pos += lab
        pr.update(incluido_fase2=1, motivo_exclusion="", n_ventanas=cnt, n_ventanas_pos=pos,
                  n_saltadas_buffer=skipped,
                  n_ventanas_esperadas_sin_buffer=int(sum(max(0, n - L - pw + 1) for L in range(ml, maxL + 1))))
        pats.append(pr)
        print(f"{key:12s} {conj:12s} PIRA={pr['pira']} ventanas={cnt:6d} pos={pos:5d}")
    return meta, seqs, pd.DataFrame(pats)


def main(df_raw=None, model=None, normalizer=None, args=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = args or load_args()
    print("Parámetros del checkpoint:", a, "| MODO:", MODO)
    if df_raw is None:
        df_raw = pd.read_csv(DATA_FILE)

    if MODO == "npz":
        npz_path = next((p for p in NPZ_CANDIDATES if p.exists()), None)
        assert npz_path is not None, f"No encuentro resultados_eval.npz en: {NPZ_CANDIDATES}"
        print("Usando", npz_path)
        z = np.load(npz_path, allow_pickle=True)
        meta, _, pac = build(df_raw, a, normalizer=None, imput="global_mean")
        w = pd.DataFrame(meta, columns=COLS)
        w = w[w.conjunto.isin(['train', 'val', 'test'])].copy()
        w['prob_pira'] = np.nan
        ok = True
        for c in ['train', 'val', 'test']:
            idx = w.index[w.conjunto == c]
            k_npz = np.asarray(z[f"{c}_window_keys"]).astype(str)
            y_npz = np.asarray(z[f"y_{c}"]).astype(int)
            k_new, y_new = w.loc[idx, 'key'].astype(str).values, w.loc[idx, 'label'].values
            if len(k_npz) != len(k_new) or (k_npz != k_new).any() or (y_npz != y_new).any():
                ok = False
                n = min(len(k_npz), len(k_new))
                diff = np.where((k_npz[:n] != k_new[:n]) | (y_npz[:n] != y_new[:n]))[0]
                print(f"❌ {c}: NO cuadra con el .npz (npz={len(k_npz)} ventanas, regeneradas={len(k_new)}; "
                      f"primera diferencia en posición {diff[0] if len(diff) else n})")
            else:
                w.loc[idx, 'prob_pira'] = np.asarray(z[f"{c}_predictions"])[:, 1]
                print(f"✅ {c}: {len(k_new)} ventanas, paciente y etiqueta idénticos al .npz")
        if not ok:
            raise SystemExit("Las ventanas no coinciden con el .npz. Cambia MODO = 'modelo' y vuelve a ejecutar.")
    else:
        model, normalizer = (model, normalizer) if model is not None else load_model()
        meta, seqs, pac = build(df_raw, a, normalizer=normalizer, imput=IMPUTACION)
        w = pd.DataFrame(meta, columns=COLS)
        w['prob_pira'] = predict_probs(model, seqs, a['seq_len'])
        w = w[w.conjunto.isin(['train', 'val', 'test'])].copy()

    w.to_csv(OUT_DIR / "ventanas_predicciones.csv", index=False)
    pac.to_csv(OUT_DIR / "pacientes.csv", index=False)
    print("\nVentanas por conjunto (size = ventanas, sum = positivas):\n", w.groupby('conjunto')['label'].agg(['size', 'sum']))
    print(f"TOTAL ventanas fase 2: {len(w)}")
    print(f"Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()
