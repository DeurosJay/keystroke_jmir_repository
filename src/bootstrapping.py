import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

# === ROOT_DIR ===
ROOT_DIR = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT_DIR)
sys.path.append(str(ROOT_DIR))

# === Cargar resultados guardados por analisis_adicionales_revision_JMIR.py ===
resultados_path = ROOT_DIR / "src" / "resultados_eval" / "resultados_eval.npz"
assert resultados_path.exists(), (
    f"No se encuentra {resultados_path}. Ejecuta primero el script principal "
    "(analisis_adicionales_revision_JMIR.py) para generar y guardar las predicciones."
)

data = np.load(resultados_path, allow_pickle=True)

y_test = data["y_test"]
test_predictions = data["test_predictions"]      # (N, 2) probabilidades softmax
test_window_keys = data["test_window_keys"]

print(f"Ventanas de test cargadas: {len(y_test)}")
print(f"Pacientes únicos en test: {len(np.unique(test_window_keys))}")
print(f"Positivos (PIRA): {int(np.sum(y_test == 1))}")


# ============================================================
# === MÉTRICAS BASE ===
# ============================================================
def compute_sensitivity(y_true, y_pred):
    """Sensibilidad = recall de la clase 1 (riesgo de PIRA)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def compute_specificity(y_true, y_pred):
    """Especificidad = recall de la clase 0 (sin riesgo de PIRA)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


# ============================================================
# === BOOTSTRAPPING POR PACIENTE ===
# ============================================================
def bootstrap_ci_by_patient(window_keys, y_true, y_pred_probs, n_bootstrap=1000, seed=42):
    """
    Remuestrea PACIENTES (no ventanas) con reemplazo, n_bootstrap veces.
    Si un paciente sale repetido en una muestra, se incluyen todas sus
    ventanas repetidas -- así se respeta que las ventanas de un mismo
    paciente están correlacionadas entre sí (no son observaciones
    independientes).

    Devuelve un dict {métrica: array de n_bootstrap valores}.
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.unique(window_keys)

    # Precalcular los índices de ventana de cada paciente una sola vez
    key_to_indices = {}
    for i, k in enumerate(window_keys):
        key_to_indices.setdefault(k, []).append(i)
    key_to_indices = {k: np.array(v) for k, v in key_to_indices.items()}

    y_pred_class = np.argmax(y_pred_probs, axis=1)
    risk_score = y_pred_probs[:, 1]

    resultados = {"sensitivity": [], "specificity": [], "accuracy": [], "auc": []}

    for _ in range(n_bootstrap):
        pacientes_muestra = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        idx = np.concatenate([key_to_indices[p] for p in pacientes_muestra])

        y_true_b = y_true[idx]
        y_pred_b = y_pred_class[idx]
        risk_b = risk_score[idx]

        resultados["sensitivity"].append(compute_sensitivity(y_true_b, y_pred_b))
        resultados["specificity"].append(compute_specificity(y_true_b, y_pred_b))
        resultados["accuracy"].append(accuracy_score(y_true_b, y_pred_b))
        try:
            resultados["auc"].append(roc_auc_score(y_true_b, risk_b))
        except ValueError:
            # p.ej. si en esa muestra bootstrap solo hay una clase presente
            pass

    return {k: np.array(v) for k, v in resultados.items()}


def print_bootstrap_summary(nombre_conjunto, resultados_bootstrap):
    n_total = len(next(iter(resultados_bootstrap.values())))
    print(f"\n===== IC 95% (bootstrap, n={n_total} remuestreos) — {nombre_conjunto} =====")
    for metrica, valores in resultados_bootstrap.items():
        valores_validos = valores[~np.isnan(valores)]
        if len(valores_validos) == 0:
            print(f"{metrica}: sin datos suficientes en las muestras bootstrap")
            continue
        media = np.mean(valores_validos)
        ic_inf = np.percentile(valores_validos, 2.5)
        ic_sup = np.percentile(valores_validos, 97.5)
        print(
            f"{metrica}: {media:.4f} (IC 95%: {ic_inf:.4f} - {ic_sup:.4f})  "
            f"[n_muestras_válidas={len(valores_validos)}/{len(valores)}]"
        )


# TODO: ajustar n_bootstrap si 1000 iteraciones tardan demasiado
N_BOOTSTRAP = 1000

bootstrap_test = bootstrap_ci_by_patient(test_window_keys, y_test, test_predictions, n_bootstrap=N_BOOTSTRAP)
print_bootstrap_summary("TEST", bootstrap_test)