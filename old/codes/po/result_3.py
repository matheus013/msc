import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Caminho da pasta com os ciclos
base_directory = "outputs"
df = extract_data_from_all_subfolders(base_directory)

# Filtrar somente técnicas Lp e sa
df = df[df["technique"].isin(["Lp", "sa"])]

# Criar boxplot com eixo X = técnica, Y = tempo, e cor = ciclo
plt.figure(figsize=(12, 6))
sns.boxplot(x="technique", y="elapsed_time_sec", hue="technique", data=df, palette="Set3")

plt.title("Distribuição do Tempo de Execução por Técnica e Ciclo")
plt.xlabel("Técnica")
plt.ylabel("Tempo de Execução (segundos)")
plt.legend(title="Ciclo", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
