#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación funcional de T1 (scheduler activo), T2 (parada por F1 macro) e I4
(inferencia con ventana rellenada a seq_len). Ejecuta el código REAL en CPU.
"""
import sys, os, glob, tempfile, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from argparse import Namespace

from timesnet.utils.tools import adjust_learning_rate
from timesnet.exp.exp_classification import Exp_Classification
from src.utils.utils import predict_model_window

# reutilizamos el generador sintético de la otra validación
from validate_normalization import build_df, FEATURES

ok = True

# ================= T1: adjust_learning_rate (cosine) =================
opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
args_lr = Namespace(lradj='cosine', learning_rate=1e-3, train_epochs=4)
lrs = []
for ep in range(1, 5):
    adjust_learning_rate(opt, ep, args_lr)
    lrs.append(opt.param_groups[0]['lr'])
expected = [1e-3/2*(1+math.cos(ep/4*math.pi)) for ep in range(1, 5)]
t1 = np.allclose(lrs, expected, atol=1e-9) and lrs[0] > lrs[-1]
print(f"[T1] scheduler cosine aplica y decrece: {t1}  lrs={['%.2e'%x for x in lrs]}")
ok &= t1

# ================= mini-entrenamiento real (T1 disparado + T2) =========
tmp = tempfile.mkdtemp()
args = Namespace(
    task_name='classification', model_id='vtrain', model='TimesNet', data='pira',
    seq_len=8, label_len=0, pred_len=0, min_seq_len=6, pred_window=3,
    pira_offset=1, buffer_days=2,
    enc_in=len(FEATURES), dec_in=len(FEATURES), c_out=2, num_class=2,
    d_model=16, d_ff=32, e_layers=1, n_heads=2, top_k=2, num_kernels=3,
    dropout=0.1, embed='timeF', freq='d', factor=1, moving_avg=25,
    activation='gelu', distil=True,
    learning_rate=5e-4, lradj='cosine', class_weight=[1.0, 5.0],
    batch_size=16, train_epochs=2, patience=5, mode='window',
    checkpoints=tmp, seed=42, num_workers=0,
    use_gpu=False, gpu=0, gpu_type='cpu', use_multi_gpu=False, devices='0',
    root_path='.', data_path='none.csv',
)

df = build_df()
exp = Exp_Classification(df.copy(), args, flag='train')
setting = "mini"

import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):          # 'Updating learning rate' va a stdout
    exp.train(setting, args)
stdout_txt = buf.getvalue()

logs = glob.glob(os.path.join(tmp, '**', 'train_log.txt'), recursive=True)
log_txt = open(logs[0]).read() if logs else ""
t2 = "F1(macro)" in log_txt
t1_fired = "Updating learning rate" in stdout_txt
ckpt = glob.glob(os.path.join(tmp, '**', 'checkpoint.pth'), recursive=True)
print(f"[T2] parada/registro por F1 macro presente en el log: {t2}")
print(f"[T1] 'Updating learning rate' disparado durante el entrenamiento: {t1_fired}")
print(f"[train] checkpoint guardado: {len(ckpt) > 0}")
ok &= t2 and t1_fired and len(ckpt) > 0

# ================= I4: inferencia con ventanas de longitud variable ====
model = exp.model
scaler = exp.scaler
args.device = torch.device('cpu')   # en producción lo fija load_gpu(args)
# df de un paciente con MÁS pasos que seq_len y con MENOS -> ambos deben funcionar
pac = df[df['key'] == df['key'].unique()[1]].copy()
try:
    for L in [args.seq_len + 20, args.seq_len, args.min_seq_len, args.min_seq_len - 2]:
        sub = pac.iloc[:max(L, 1)]
        pred, p0, p1 = predict_model_window(model, args, sub.copy(), scaler, seq_len=None)
        assert pred in (0, 1) and np.isfinite(p1).all(), f"salida inválida en L={L}"
    i4 = True
except Exception as e:
    i4 = False
    print(f"[I4] excepción: {type(e).__name__}: {e}")
print(f"[I4] predict_model_window funciona con longitudes variables (rellena a seq_len): {i4}")
ok &= i4

print("\n" + ("✅ T1/T2/I4 VALIDADOS" if ok else "❌ ALGUNA VALIDACIÓN FALLA"))
sys.exit(0 if ok else 1)
