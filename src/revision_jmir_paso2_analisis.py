"""
REVISIÓN JMIR (2.ª ronda) — PASO 2: todas las cifras para la respuesta al editor.
No necesita ni el modelo ni GPU: lee lo que guardó el paso 1.

Genera en OUT_DIR:
  resultados_revision_jmir.xlsx   (una hoja por punto del editor)
  resumen_revision_jmir.txt       (lo esencial, para copiar a la carta)
  fig5_por_longitud_test.png      (sustituta de la Figura 5)
  figS_prob_vs_dias_al_evento.png (Mandatory 8)

Hojas del Excel:
  M1_particion        nº pacientes / pacientes PIRA / ventanas por conjunto + comprobación de solapamiento
  M3_metricas         sens, esp, PPV, NPV, exactitud, exactitud balanceada, AUROC, AUPRC, prevalencia
                      (train / val / test; Recommended 5) con IC 95 % por bootstrap de pacientes
  M2_por_paciente     alerta sí/no, primera alerta, antelación (días antes de la detección clínica)
  M4_por_longitud     métricas del test por longitud de entrada (60–90)
  M8_estatico         modelo logístico solo con variables estáticas + TS basal, mismo split
  M8_negativas_PIRA   especificidad en ventanas negativas de pacientes PIRA vs no PIRA
  M10_recuentos       ventanas generadas vs esperadas por paciente
  R1_exclusiones      pacientes fuera de la fase 2 y motivo
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Colócalo en ...\emcare-ml-fix-git\src (igual que el paso 1); lee lo que dejó el paso 1.
IN_DIR = Path(__file__).resolve().parent / "resultados_revision_jmir"
OUT_DIR = IN_DIR
UMBRAL = 0.5          # argmax de softmax con 2 clases = prob_pira >= 0.5
N_BOOT = 1000
SEED = 42
DIAS_PERSISTENCIA = 7  # regla alternativa de alerta: >= 7 días consecutivos con alerta


# ------------------------------------------------------------------ métricas
def _conf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    return tp, fn, tn, fp


def metricas(y, s, umbral=UMBRAL):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); p = (s >= umbral).astype(int)
    tp, fn, tn, fp = _conf(y, p)
    sens = tp / (tp + fn) if tp + fn else np.nan
    esp = tn / (tn + fp) if tn + fp else np.nan
    two = len(np.unique(y)) == 2
    return dict(n_ventanas=len(y), n_pos=int(y.sum()), prevalencia=y.mean() if len(y) else np.nan,
                TP=tp, FN=fn, TN=tn, FP=fp, sensibilidad=sens, especificidad=esp,
                PPV=tp / (tp + fp) if tp + fp else np.nan, NPV=tn / (tn + fn) if tn + fn else np.nan,
                exactitud=(tp + tn) / len(y) if len(y) else np.nan,
                exactitud_balanceada=np.nanmean([sens, esp]),
                F1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                AUROC=roc_auc_score(y, s) if two else np.nan,
                AUPRC=average_precision_score(y, s) if two else np.nan,
                exactitud_clase_mayoritaria=max(y.mean(), 1 - y.mean()) if len(y) else np.nan)


def bootstrap_paciente(df, col_score="prob_pira", n=None, seed=SEED):
    n = n or N_BOOT
    """Remuestrea PACIENTES con reemplazo (como bootstrapping.py). Devuelve IC 2,5–97,5 % y nº de
    iteraciones válidas por métrica. Se informa la estimación PUNTUAL aparte (no la media bootstrap)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby('key')}
    keys = np.array(list(groups))
    res = {m: [] for m in ['sensibilidad', 'especificidad', 'exactitud', 'exactitud_balanceada', 'AUROC', 'AUPRC', 'PPV']}
    for _ in range(n):
        smp = pd.concat([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        m = metricas(smp['label'].values, smp[col_score].values)
        for k in res:
            res[k].append(m[k])
    out = {}
    for k, v in res.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        out[k] = (np.percentile(v, 2.5) if len(v) else np.nan,
                  np.percentile(v, 97.5) if len(v) else np.nan, len(v))
    return out


# ------------------------------------------------------------------ main
def main():
    w = pd.read_csv(IN_DIR / "ventanas_predicciones.csv",
                    parse_dates=['fecha_inicio_input', 'fecha_fin_input', 'fecha_inicio_horizonte', 'fecha_fin_horizonte'])
    pac = pd.read_csv(IN_DIR / "pacientes.csv", parse_dates=['fecha_cdp', 'primer_dia', 'ultimo_dia'])
    w['pred'] = (w['prob_pira'] >= UMBRAL).astype(int)
    sheets, txt = {}, []

    # ---------- M1: partición
    if (pac.conjunto == 'no_asignado').any():
        txt.append(f"⚠️ Pacientes con datos suficientes pero sin conjunto: {list(pac.loc[pac.conjunto == 'no_asignado', 'key'])}")
    m1 = (pac[pac.incluido_fase2 == 1].groupby('conjunto')
          .agg(pacientes=('key', 'size'), pacientes_PIRA=('pira', 'sum')).reindex(['train', 'val', 'test']))
    wc = w.groupby('conjunto')['label'].agg(ventanas='size', ventanas_PIRA='sum')
    m1 = m1.join(wc)
    m1['%_ventanas_del_total'] = 100 * m1['ventanas'] / m1['ventanas'].sum()
    m1.loc['TOTAL'] = m1.sum(numeric_only=True)
    sets = {c: set(w.loc[w.conjunto == c, 'key']) for c in ['train', 'val', 'test']}
    solap = {f"{a}∩{b}": len(sets[a] & sets[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    sheets['M1_particion'] = m1.reset_index()
    txt += ["== MANDATORY 1: partición por paciente (fase 2)", m1.round(1).to_string(),
            f"Pacientes compartidos entre conjuntos: {solap}  (deben ser 0)",
            f"Pacientes del CSV: {len(pac)} | incluidos en fase 2: {int(pac.incluido_fase2.sum())} | "
            f"PIRA incluidos: {int(pac.loc[pac.incluido_fase2 == 1, 'pira'].sum())} de {int(pac.pira.sum())}", ""]

    # ---------- M3 + R5: métricas por conjunto con IC
    rows = []
    for c in ['train', 'val', 'test']:
        d = w[w.conjunto == c]
        m = metricas(d['label'], d['prob_pira'])
        ci = bootstrap_paciente(d) if c != 'train' else {}
        row = {'conjunto': c, 'pacientes': d.key.nunique(), 'pacientes_PIRA': d.loc[d.label == 1, 'key'].nunique(), **m}
        for k, (lo, hi, nv) in ci.items():
            row[f'{k}_IC95'] = f"{lo:.3f}–{hi:.3f} (iter. válidas {nv}/{N_BOOT}; una iteración sin pacientes PIRA no cuenta para sens/AUROC/AUPRC)"
        rows.append(row)
    m3 = pd.DataFrame(rows)
    sheets['M3_metricas'] = m3
    t = m3.set_index('conjunto').T
    txt += ["== MANDATORY 3 / RECOMMENDED 5: métricas (estimación puntual; IC por bootstrap de pacientes)",
            t.to_string(), ""]

    # ---------- M2: por paciente (val y test) — alertas diarias
    # Puntuación diaria = media de prob_pira de todas las ventanas (todas las longitudes) que terminan ese día.
    daily = (w[w.conjunto.isin(['val', 'test'])]
             .groupby(['conjunto', 'key', 'fecha_fin_input'])
             .agg(score=('prob_pira', 'mean'), label=('label', 'max')).reset_index())
    daily['alerta'] = (daily['score'] >= UMBRAL).astype(int)
    prow = []
    for (c, k), g in daily.groupby(['conjunto', 'key']):
        g = g.sort_values('fecha_fin_input')
        info = pac.set_index('key').loc[k]
        cdp = info['fecha_cdp']
        # racha de días consecutivos (en filas) con alerta
        run, best, first_persist = 0, 0, pd.NaT
        for d_, a in zip(g['fecha_fin_input'], g['alerta']):
            run = run + 1 if a else 0
            if run >= DIAS_PERSISTENCIA and pd.isna(first_persist):
                first_persist = d_
            best = max(best, run)
        first = g.loc[g.alerta == 1, 'fecha_fin_input'].min()
        r = dict(conjunto=c, key=k, PIRA=int(info['pira']), fecha_deteccion_clinica=cdp,
                 dias_evaluados=len(g), dias_con_alerta=int(g.alerta.sum()),
                 pct_dias_con_alerta=round(100 * g.alerta.mean(), 1),
                 alerta_alguna_vez=int(g.alerta.any()), racha_max_dias_alerta=best,
                 primera_alerta=first, primera_alerta_persistente=first_persist,
                 falsas_alertas_por_paciente_mes=np.nan, antelacion_dias=np.nan, antelacion_persistente_dias=np.nan,
                 score_medio_periodo_positivo=g.loc[g.label == 1, 'score'].mean(),
                 score_medio_periodo_negativo=g.loc[g.label == 0, 'score'].mean())
        if info['pira'] == 1:
            r['antelacion_dias'] = (cdp - first).days if pd.notna(first) else np.nan
            r['antelacion_persistente_dias'] = (cdp - first_persist).days if pd.notna(first_persist) else np.nan
        else:
            r['falsas_alertas_por_paciente_mes'] = round(g.alerta.sum() / (len(g) / 30.4), 2)
        prow.append(r)
    m2 = pd.DataFrame(prow).sort_values(['conjunto', 'PIRA', 'key'], ascending=[False, False, True])
    sheets['M2_por_paciente'] = m2
    tt = m2[m2.conjunto == 'test']
    txt += ["== MANDATORY 2: resultados por paciente (TEST)",
            f"Pacientes PIRA en test: {int(tt.PIRA.sum())} de {len(tt)}",
            tt[['key', 'PIRA', 'alerta_alguna_vez', 'antelacion_dias', 'antelacion_persistente_dias',
                'pct_dias_con_alerta', 'falsas_alertas_por_paciente_mes']].to_string(index=False),
            f"No-PIRA del test con >=1 alerta: {int(tt.loc[tt.PIRA == 0, 'alerta_alguna_vez'].sum())} de {int((tt.PIRA == 0).sum())}",
            "OJO: con datos diarios, en un paciente PIRA la 'alerta' puede aparecer ya en el periodo negativo; "
            "mira score_medio_periodo_positivo vs score_medio_periodo_negativo.", ""]

    # ---------- M4: por longitud en test
    te = w[w.conjunto == 'test']
    m4 = pd.DataFrame([{'longitud': L, **metricas(g['label'], g['prob_pira'])} for L, g in te.groupby('longitud')])
    sheets['M4_por_longitud'] = m4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, lab in [('sensibilidad', 'Sensitivity'), ('especificidad', 'Specificity'), ('exactitud_balanceada', 'Balanced accuracy')]:
        ax.plot(m4['longitud'], m4[col], marker='o', ms=3, label=lab)
    ax.set_xlabel('Input horizon (days)'); ax.set_ylabel('Value'); ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3); ax.legend(frameon=False)
    ax.set_title('Phase 2 test set: performance by input horizon length')
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig5_por_longitud_test.png", dpi=300); plt.close(fig)
    txt += ["== MANDATORY 4: figura 5 nueva -> fig5_por_longitud_test.png (test real, mismas ventanas que M3)", ""]

    # ---------- M8a: probabilidad vs días hasta el evento (pacientes PIRA)
    pw = w[w.dias_fin_input_a_cdp.notna()].copy()
    pw['bin'] = (pw['dias_fin_input_a_cdp'] // 10) * 10
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, mk in [('train', 'o'), ('val', 's'), ('test', '^')]:
        d = pw[pw.conjunto == c].groupby('bin')['prob_pira'].mean()
        if len(d):
            ax.plot(d.index, d.values, marker=mk, ms=3, label=f'{c} (n={pw[pw.conjunto == c].key.nunique()} PIRA patients)')
    ax.axvspan(31, 151, color='grey', alpha=.12, label='Positive-label interval')
    ax.invert_xaxis(); ax.set_xlabel('Days from end of input window to first clinical detection of PIRA')
    ax.set_ylabel('Mean predicted probability'); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "figS_prob_vs_dias_al_evento.png", dpi=300); plt.close(fig)

    # ---------- M8b: especificidad en negativas de pacientes PIRA vs no-PIRA
    piras = set(pac.loc[pac.pira == 1, 'key'])
    neg = w[w.label == 0].copy(); neg['paciente_PIRA'] = neg.key.isin(piras).astype(int)
    m8b = (neg.groupby(['conjunto', 'paciente_PIRA'])
           .agg(ventanas=('pred', 'size'), pacientes=('key', 'nunique'), especificidad=('pred', lambda x: 1 - x.mean()),
                prob_media=('prob_pira', 'mean')).reset_index())
    sheets['M8_negativas_PIRA'] = m8b
    txt += ["== MANDATORY 8b: especificidad en ventanas NEGATIVAS según si el paciente acaba teniendo PIRA",
            m8b.round(3).to_string(index=False),
            "Si en pacientes PIRA la especificidad es mucho menor que en no-PIRA -> el modelo reconoce al paciente.", ""]

    # ---------- M8c: modelo estático (logística) con el MISMO split
    stat_cols = ['ts_basal_30d', 'Edad', 'Genero', 'FAME_n_High', 'FAME_n_Low', 'Tipo_EM_EMPP', 'Tipo_EM_EMSP']
    if 'PST' in pac.columns: stat_cols.append('PST')
    elif 'PST_basal' in pac.columns: stat_cols.append('PST_basal')
    ws = w.merge(pac[['key'] + stat_cols], on='key', how='left')
    ws[stat_cols] = ws[stat_cols].fillna(ws.loc[ws.conjunto == 'train', stat_cols].mean())
    tr = ws[ws.conjunto == 'train']
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000))
    lr.fit(tr[stat_cols], tr['label'])
    rows = []
    for c in ['val', 'test']:
        d = ws[ws.conjunto == c].copy()
        d['prob_estatico'] = lr.predict_proba(d[stat_cols])[:, 1]
        rows.append({'conjunto': c, 'modelo': 'Logístico estático (' + ', '.join(stat_cols) + ')',
                     **metricas(d['label'], d['prob_estatico'])})
        rows.append({'conjunto': c, 'modelo': 'TimesNet', **metricas(d['label'], d['prob_pira'])})
    m8c = pd.DataFrame(rows)
    sheets['M8_estatico'] = m8c
    txt += ["== MANDATORY 8c: TimesNet vs modelo solo estático (mismas ventanas, mismo split)",
            m8c[['conjunto', 'modelo', 'AUROC', 'AUPRC', 'exactitud_balanceada', 'sensibilidad', 'especificidad']]
            .round(3).to_string(index=False),
            "Nota: el modelo estático da la MISMA probabilidad a todas las ventanas de un paciente.", ""]

    # ---------- M10 / R1
    inc = pac[pac.incluido_fase2 == 1]
    sheets['M10_recuentos'] = inc[['key', 'conjunto', 'pira', 'n_dias_con_datos', 'dias_calendario', 'pct_dias_sin_datos',
                                   'n_ventanas', 'n_ventanas_pos', 'n_saltadas_buffer', 'n_ventanas_esperadas_sin_buffer']]
    sheets['R1_exclusiones'] = pac[pac.incluido_fase2 == 0][['key', 'pira', 'n_dias_con_datos', 'motivo_exclusion']]
    txt += ["== MANDATORY 10 / RECOMMENDED 1",
            f"Ventanas generadas: {int(inc.n_ventanas.sum())} | sin la regla de buffer serían "
            f"{int(inc.n_ventanas_esperadas_sin_buffer.sum())} | descartadas por buffer/post-evento: {int(inc.n_saltadas_buffer.sum())}",
            f"Excluidos de fase 2 por pocos días: {int((pac.incluido_fase2 == 0).sum())} "
            f"(PIRA: {int(pac.loc[pac.incluido_fase2 == 0, 'pira'].sum())})",
            f"% medio de días de calendario sin datos (incluidos): {inc.pct_dias_sin_datos.mean():.1f}%",
            f"¿FAME cambia dentro de algún paciente? {int(pac.fame_cambia_en_el_tiempo.sum())} pacientes (debe ser 0 -> DMT estático)", ""]

    with pd.ExcelWriter(OUT_DIR / "resultados_revision_jmir.xlsx") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    (OUT_DIR / "resumen_revision_jmir.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print(f"\nGuardado en {OUT_DIR}")


if __name__ == "__main__":
    main()
