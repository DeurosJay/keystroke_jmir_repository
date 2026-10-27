# Smartphone keystroke dynamics and PIRA in multiple sclerosis — analysis code

Code accompanying the manuscript *"Smartphone Keystroke Dynamics and Progression Independent of Relapse Activity in Multiple Sclerosis: Exploratory Prospective Cohort Study Using Machine Learning"* (JMIR, under review).

The repository contains the code used for window generation, labelling, patient-level partitioning, model training and evaluation, and the additional analyses performed during peer review. **No patient data are included.** Keystroke data are protected and the clinical dataset is available from the corresponding author upon reasonable request (see the Data availability statement of the manuscript).

## Structure

```
timesnet/        TimesNet model, data loaders, window generation and labelling
                 (exp/exp_classification.py), training entry point (run.py) and
                 hyperparameter grid search (grid.py)
src/
  contaje_eventos.py                    window regeneration, labelling and evaluation of the final model
  bootstrapping.py                      patient-level bootstrap confidence intervals
  revision_jmir_paso1_predicciones.py   per-window predictions for training/validation/test sets
  revision_jmir_paso2_analisis.py       window- and patient-level metrics, metrics by input length
  figura5_auroc_por_longitud.py         Figure 4 (AUROC by input window length, test set)
  analisis_adicionales_revision_JMIR.py additional evaluation analyses
  database/mongo.py                     data extraction (credentials read from environment variables)
  utils/                                inference utilities
modelos/         phase 1 exploratory models (XGBoost and Transformer notebooks)
tests/           validation scripts on synthetic data
```

## Labelling (phase 2, final configuration)

For patient *i*, let *E(i)* be the date of first clinical detection of the PIRA event. Input windows of 60–90 days with keystroke data end on day *t*; the prediction horizon spans the following 120 days of data. A window is labelled positive if *E(i) − 150 ≤ t+1 ≤ E(i) − 30*; windows whose horizon ends later than *E(i) − 35* are excluded; all other windows, including all windows from patients without PIRA, are labelled negative. Windows are built on consecutive days with data, with a stride of one day. Patients are allocated to training, validation, and test sets (stratified by PIRA status), and all windows of a given patient belong to the same set.

## Requirements

```bash
pip install -r requirements.txt            # core
pip install -r requirements-optional.txt   # database access and optional models
```

Database access is configured through environment variables (`MONGO_HOST`, `MONGO_PORT`, `MONGO_USER`, `MONGO_PASSWORD`, `MONGO_DATABASE`, and `MONGO_SSH_*` if an SSH tunnel is used). Input/output paths are set at the top of each script.

## Contact

Juan Luis Chico-García — Department of Neurology, Hospital Universitario Ramón y Cajal, Madrid, Spain.
