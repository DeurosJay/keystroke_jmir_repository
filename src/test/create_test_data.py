import pandas as pd

# Cargar el CSV del usuario
csv_path = "../merged_daily.csv"
df = pd.read_csv(csv_path)

seq_len = 90
pred_window = 15

df['date'] = pd.to_datetime(df['date'])
df['Date_CDP'] = pd.to_datetime(df['Date_CDP'], errors='coerce')

# Escoge un paciente único (cambia esto si quieres otro)
unique_keys = df['key'].unique()

# Inicializar flags
found_1 = False
found_0 = False

# Buscar en todos los pacientes hasta encontrar ambos
for key in df['key'].unique():
    df_patient = df[df['key'] == key].sort_values('date').reset_index(drop=True)
    pira_date = df_patient['Date_CDP'].dropna().unique()
    pira_date = pd.to_datetime(pira_date[0]) if len(pira_date) > 0 else None

    # Buscar label = 1
    if not found_1 and pira_date is not None:
        for i in range(len(df_patient) - seq_len - pred_window + 1):
            window = df_patient.iloc[i:i + seq_len]
            future = df_patient.iloc[i + seq_len:i + seq_len + pred_window]
            if future['date'].min() <= pira_date <= future['date'].max():
                window.to_csv("sample_pira_1.csv", index=False)
                print(f"✅ sample_pira_1.csv guardado (paciente {key})")
                found_1 = True
                break

    # Buscar label = 0
    if not found_0:
        for i in range(len(df_patient) - seq_len - pred_window + 1):
            window = df_patient.iloc[i:i + seq_len]
            future = df_patient.iloc[i + seq_len:i + seq_len + pred_window]
            if not pira_date or not (future['date'].min() <= pira_date <= future['date'].max()):
                window.to_csv("sample_pira_0.csv", index=False)
                print(f"✅ sample_pira_0.csv guardado (paciente {key})")
                found_0 = True
                break

    if found_1 and found_0:
        break