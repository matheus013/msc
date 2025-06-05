import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def extract_data_from_all_subfolders(base_directory):
    all_data = []

    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # pula se não for pasta

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


# Exemplo de uso
base_directory = "outputs"
df = extract_data_from_all_subfolders(base_directory)
df = df[df['technique'] != "ga"]
# Agrupamento por ciclo e técnica para calcular média de tempo
agg_df = df.groupby(["cycle", "technique"])["elapsed_time_sec"].mean().reset_index()

# Plotagem do gráfico
plt.figure(figsize=(10, 6))

for technique in agg_df["technique"].unique():
    subset = agg_df[agg_df["technique"] == technique]
    plt.plot(subset["cycle"], subset["elapsed_time_sec"], marker='o', label=technique)

plt.xlabel("Cycle")
plt.ylabel("Average Elapsed Time (sec)")
plt.title("Comparative Execution Time per Cycle by Technique")
plt.legend(title="Technique")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()