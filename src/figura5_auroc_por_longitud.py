"""
Figura 5 (revisión JMIR): AUROC por longitud de ventana de entrada en TEST (descriptiva).

Entrada: ventanas_predicciones.csv (salida de revision_jmir_paso1_predicciones.py),
         columnas usadas: key, conjunto, longitud, label, prob_pira
Salida:  figura5_auroc_por_longitud.png / .tiff y auroc_por_longitud_test.csv

Uso:  python figura5_auroc_por_longitud.py
Requisitos: pandas, numpy, scikit-learn, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================ CONFIG ============================
ENTRADA = Path("resultados_revision_jmir/ventanas_predicciones.csv")
SALIDA = ENTRADA.parent
MOSTRAR_IC = False     # True = banda IC 95% por bootstrap de pacientes
N_BOOT = 1000
SEMILLA = 42
# ================================================================

w = pd.read_csv(ENTRADA, usecols=["key", "conjunto", "longitud", "label", "prob_pira"])
d = w[w["conjunto"] == "test"]
rng = np.random.default_rng(SEMILLA)

filas = []
for L, g in d.groupby("longitud"):
    y, s = g["label"].values, g["prob_pira"].values
    fila = dict(longitud=L, n_ventanas=len(g), n_positivas=int(y.sum()),
                AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s),
                prevalencia=y.mean())
    if MOSTRAR_IC:
        # Bootstrap por paciente: se remuestrean pacientes (con todas sus ventanas)
        grupos = {k: gg for k, gg in g.groupby("key")}
        claves = list(grupos)
        bs = []
        for _ in range(N_BOOT):
            m = pd.concat([grupos[k] for k in rng.choice(claves, len(claves), replace=True)])
            if m["label"].nunique() == 2:          # se descartan remuestras sin positivos
                bs.append(roc_auc_score(m["label"], m["prob_pira"]))
        fila.update(IC_inf=np.percentile(bs, 2.5), IC_sup=np.percentile(bs, 97.5), iter_validas=len(bs))
    filas.append(fila)

r = pd.DataFrame(filas)
r.to_csv(SALIDA / "auroc_por_longitud_test.csv", index=False)
print(r.round(3).to_string(index=False))

# ---------------------------- figura ----------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
if MOSTRAR_IC:
    ax.fill_between(r["longitud"], r["IC_inf"], r["IC_sup"], color="#2a78d6", alpha=.18, lw=0)
ax.plot(r["longitud"], r["AUROC"], color="#2a78d6", lw=2, marker="o", ms=3)
ax.axhline(.5, color="#8a8985", lw=1, ls=":")
ax.text(60.5, .51, "chance", fontsize=8, color="#52514e", va="bottom")
ax.set_xlim(60, 90)
ax.set_ylim(0.4 if not MOSTRAR_IC else 0, 1.0)
ax.set_xlabel("Input window length (days with typing data)")
ax.set_ylabel("AUROC (test set)")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#8a8985")
ax.grid(axis="y", color="#e6e5e0", lw=.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(SALIDA / "figura5_auroc_por_longitud.png")
fig.savefig(SALIDA / "figura5_auroc_por_longitud.tiff")
print("Figura guardada en", SALIDA)
