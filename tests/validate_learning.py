#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación CONDUCTUAL (opción A): confirmar que la arquitectura TimesNet, con el
pipeline corregido, APRENDE de extremo a extremo — no solo que corre.

Se genera un dataset sintético con una SEÑAL FUERTE y aprendible (los pacientes
con evento PIRA presentan un ascenso claro en las features antes del evento) y se
entrena el modelo real a través de create_dataset -> train -> test. Si el AUC de
test es alto, la red aprende. No prueba nada clínico; prueba capacidad de aprendizaje.
"""
import sys, os, tempfile, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from timesnet.exp.exp_classification import Exp_Classification

FEATURES = [
    'mean_added_count', 'median_added_count', 'max_added_count', 'min_added_count',
    'mean_removed_count', 'median_removed_count', 'max_removed_count', 'min_removed_count',
    'mean_index', 'median_index', 'max_index', 'min_index',
    'Edad', 'Genero', 'FAME_n_High', 'FAME_n_Low', 'FAME_n_No',
    'Tipo_EM_EMPP', 'Tipo_EM_EMRR', 'Tipo_EM_EMSP'
]


def make_learnable(n_patients=80, seed=7, signal=4.0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        positive = (p % 2 == 0)
        n_days = int(rng.integers(150, 220))
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        base = rng.normal(0, 1, size=(n_days, len(FEATURES))).astype(np.float32)
        if positive:
            cdp = int(rng.integers(int(n_days * 0.65), n_days - 5))
            date_cdp = dates[cdp]
            # SEÑAL: ascenso fuerte en las 12 features de conteo en los ~50 días previos
            lo = max(0, cdp - 50)
            base[lo:cdp + 1, :12] += signal
        else:
            date_cdp = pd.NaT
        for i in range(n_days):
            row = {'key': f"P{p:03d}", 'date': str(dates[i]),
                   'Date_CDP': (str(date_cdp) if positive else '')}
            for j, f in enumerate(FEATURES):
                row[f] = float(base[i, j])
            row['Edad'] = float(40 + p % 30); row['Genero'] = float(p % 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    df['Date_CDP'] = df['Date_CDP'].replace('', np.nan)
    return df


def main():
    tmp = tempfile.mkdtemp()
    args = Namespace(
        task_name='classification', model_id='learn', model='TimesNet', data='pira',
        seq_len=90, label_len=0, pred_len=0, min_seq_len=90, pred_window=30,
        pira_offset=10, buffer_days=5, window_stride=10,
        enc_in=len(FEATURES), dec_in=len(FEATURES), c_out=2, num_class=2,
        d_model=32, d_ff=64, e_layers=2, n_heads=2, top_k=3, num_kernels=4,
        dropout=0.1, embed='timeF', freq='d', factor=1, moving_avg=25,
        activation='gelu', distil=True,
        learning_rate=1e-3, lradj='cosine', class_weight=[1.0, 3.0],
        batch_size=64, train_epochs=18, patience=8, mode='window',
        checkpoints=tmp, seed=42, num_workers=0,
        use_gpu=False, gpu=0, gpu_type='cpu', use_multi_gpu=False, devices='0',
        root_path='.', data_path='none.csv',
    )

    print("Generando dataset sintético con señal aprendible…")
    df = make_learnable()
    exp = Exp_Classification(df.copy(), args, flag='train')
    print(f"Ventanas: train={len(exp.X_train)} val={len(exp.X_val)} test={len(exp.X_test)} "
          f"| positivos train={int(np.sum(exp.y_train))}")

    print("Entrenando (pipeline corregido, CPU)…")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exp.train('learn', args)

    # Evaluación en TEST con el mejor checkpoint
    crit = exp._select_criterion(args.class_weight, use_focal=True)
    test_data, test_loader = exp._get_data('test', exp.X_test, exp.y_test)
    _, acc, preds, trues, p0, p1 = exp.vali(test_data, test_loader, crit)
    trues = np.asarray(trues).astype(int)

    auc = roc_auc_score(trues, p1) if len(np.unique(trues)) > 1 else float('nan')
    ap = average_precision_score(trues, p1) if len(np.unique(trues)) > 1 else float('nan')
    pred_lbl = (p1 >= 0.5).astype(int)
    cm = confusion_matrix(trues, pred_lbl)

    print("\n──────── Resultado en TEST (split por paciente) ────────")
    print(f"  n_test={len(trues)}  positivos={int(trues.sum())}")
    print(f"  Accuracy   : {acc:.3f}")
    print(f"  ROC-AUC    : {auc:.3f}")
    print(f"  PR-AUC     : {ap:.3f}")
    print(f"  Matriz confusión [fila=real, col=pred]:\n           pred0  pred1")
    for i, rrow in enumerate(cm):
        print(f"    real{i}   " + "  ".join(f"{v:5d}" for v in rrow))

    passed = (auc >= 0.85)
    print("\n" + ("✅ LA RED APRENDE (AUC de test ≥ 0.85): arquitectura + pipeline corregido funcionan end-to-end."
                  if passed else
                  "❌ AUC de test < 0.85: revisar (con esta señal fuerte debería aprender)."))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
