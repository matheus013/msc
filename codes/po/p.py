import os
import json
import pandas as pd
from scipy.stats import ttest_ind

def extract_data_from_all_subfolders(base_directory):
    all_data = []

    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            if filename.startswith("solution_") and filename.endswith(".json"):
                filepath = os.path.join(subfolder_path, filename)
                technique = filename.split("_")[1].replace(".json", "")
                if technique.lower() == "run":
                    technique = "Lp"

                with open(filepath, "r") as f:
                    content = json.load(f)

                value = content.get("objective_value") or content.get("cost")

                row = {
                    "cycle": subfolder,
                    "technique": technique,
                    "run": content.get("run"),
                    "elapsed_time_sec": content.get("elapsed_time_sec"),
                    "value": value
                }

                all_data.append(row)

    return pd.DataFrame(all_data)

# Caminho para os dados
base_directory = "outputs"
df = extract_data_from_all_subfolders(base_directory)

# Filtra somente as técnicas Lp e sa
df_lp = df[df["technique"] == "Lp"]
df_sa = df[df["technique"] == "sa"]

# Verifica se existem dados suficientes
if df_lp.empty or df_sa.empty:
    print("❌ Dados insuficientes para Lp ou sa.")
else:
    # Teste de hipótese: Welch's t-test (assume variâncias diferentes)
    stat, p_value = ttest_ind(df_lp["elapsed_time_sec"], df_sa["elapsed_time_sec"], equal_var=False)

    print("✅ Teste de hipótese: diferença de tempo entre Lp e sa")
    print(f"Estatística t: {stat:.4f}")
    print(f"Valor-p: {p_value:.4e}")

    alpha = 0.05
    if p_value < alpha:
        print("📌 Diferença significativa detectada (p < 0.05)")
    else:
        print("📎 Nenhuma diferença estatisticamente significativa (p >= 0.05)")
