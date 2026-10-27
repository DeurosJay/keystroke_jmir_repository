# -*- coding: utf-8 -*-
"""
[FIX B2] Fuente única de verdad para el conjunto de features del modelo EMPIRA.

Antes esta lista de 20 columnas estaba duplicada literalmente en varios ficheros
(experimento, predictor, utilidades) y una copia desactualizada de 12 columnas en
data_loader.py, lo que ya había provocado divergencias (D3). Importar desde aquí
evita que se vuelvan a desincronizar.
"""

PIRA_FEATURES = [
    'mean_added_count', 'median_added_count', 'max_added_count', 'min_added_count',
    'mean_removed_count', 'median_removed_count', 'max_removed_count', 'min_removed_count',
    'mean_index', 'median_index', 'max_index', 'min_index',
    'Edad', 'Genero', 'FAME_n_High', 'FAME_n_Low', 'FAME_n_No',
    'Tipo_EM_EMPP', 'Tipo_EM_EMRR', 'Tipo_EM_EMSP',
]
