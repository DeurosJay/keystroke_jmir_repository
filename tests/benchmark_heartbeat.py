#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducción de benchmark del paper TimesNet — dataset UEA 'Heartbeat'.

Alimenta el Model de TimesNet (el de este repo, tras la auditoría) DIRECTAMENTE
con el dataset del paper, sin pasar por create_dataset (el repo reemplazó el
loader UEA original). Objetivo: confirmar que la ARQUITECTURA reproduce el orden
de magnitud publicado (TimesNet ≈ 77.6% accuracy en Heartbeat).

No valida las correcciones del pipeline de EMPIRA (eso son datos reales, A2/A3).
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
import torch, torch.nn as nn
from argparse import Namespace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from aeon.datasets import load_classification
from timesnet.models.TimesNet import Model

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(max(1, os.cpu_count() - 1))
PUBLISHED = 0.776  # TimesNet, Heartbeat (paper / Time-Series-Library)


def load_split(name='Heartbeat'):
    Xtr, ytr = load_classification(name, split='train')
    Xte, yte = load_classification(name, split='test')
    # (n, C, T) -> (n, T, C)
    Xtr = np.transpose(Xtr, (0, 2, 1)).astype(np.float32)
    Xte = np.transpose(Xte, (0, 2, 1)).astype(np.float32)
    le = LabelEncoder().fit(ytr)
    ytr = le.transform(ytr); yte = le.transform(yte)
    # estandarización por canal con estadísticos de TRAIN (sin fuga)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0); std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std; Xte = (Xte - mean) / std
    return Xtr, ytr, Xte, yte


def make_model(seq_len, enc_in, num_class):
    # Config alineada con Time-Series-Library (clasificación UEA).
    cfg = Namespace(task_name='classification', seq_len=seq_len, label_len=0, pred_len=0,
                    e_layers=2, enc_in=enc_in, d_model=64, embed='timeF', freq='h',
                    dropout=0.1, num_kernels=6, d_ff=128, top_k=3, num_class=num_class, c_out=num_class)
    return Model(cfg)


def batches(X, y, bs, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    Xt = torch.tensor(X, device=device)
    mask = torch.ones((len(X), X.shape[1]), dtype=torch.float32, device=device)
    logits = model(Xt, mask, None, None)
    p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return pred, p1


def main():
    device = torch.device('cpu')
    Xtr_all, ytr_all, Xte, yte = load_split('Heartbeat')
    n, T, C = Xtr_all.shape
    print(f"Heartbeat: train={n} test={len(Xte)} | T={T} canales={C} | clases={np.bincount(ytr_all).tolist()}")
    maj = max(np.bincount(yte)) / len(yte)
    print(f"Baseline clase mayoritaria (test): {maj:.3f}  ·  publicado TimesNet: {PUBLISHED:.3f}")

    # Protocolo del paper (Time-Series-Library): se entrena con TODO el train y se
    # monitoriza el test cada época, reportando la mejor accuracy de test. No es
    # generalización held-out; es reproducción del número publicado con su misma
    # metodología. CrossEntropy plana (la métrica publicada es accuracy).
    model = make_model(T, C, 2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc, best_auc, patience, bad = -1, 0, 12, 0
    t0 = time.time()
    for epoch in range(1, 101):
        model.train()
        for xb, yb in batches(Xtr_all, ytr_all, bs=16):
            opt.zero_grad()
            xt = torch.tensor(xb, device=device)
            mask = torch.ones((len(xb), T), dtype=torch.float32, device=device)
            loss = crit(model(xt, mask, None, None), torch.tensor(yb, device=device).long())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 4.0); opt.step()
        pred, p1 = evaluate(model, Xte, yte, device)
        acc = accuracy_score(yte, pred); auc = roc_auc_score(yte, p1)
        if acc > best_acc:
            best_acc, best_auc, bad = acc, auc, 0
        else:
            bad += 1
        if epoch % 5 == 0 or bad == 0:
            print(f"  epoch {epoch:3d} | test_acc={acc:.3f} auc={auc:.3f} | best_acc={best_acc:.3f} | {time.time()-t0:.0f}s", flush=True)
        if bad >= patience:
            print(f"  early stopping en epoch {epoch}"); break

    acc, auc = best_acc, best_auc
    print("\n──────── TEST (Heartbeat) — mejor época ────────")
    print(f"  Accuracy : {acc:.3f}   (publicado ≈ {PUBLISHED:.3f}, mayoría {maj:.3f})")
    print(f"  ROC-AUC  : {auc:.3f}")

    reproduce = (acc >= PUBLISHED - 0.05) and (auc >= 0.65)
    print("\n" + ("✅ REPRODUCE el orden de magnitud publicado (accuracy en rango y AUC > mayoría)."
                  if reproduce else
                  "⚠ Por debajo del rango publicado con esta config ligera de CPU (ver nota)."))
    sys.exit(0 if reproduce else 1)


if __name__ == "__main__":
    main()
