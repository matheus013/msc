import os
import json
import pandas as pd

def extract_data_from_jsons(directory):
    data = []

    for filename in os.listdir(directory):
        if filename.startswith("solution_") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            technique = filename.split("_")[1].replace(".json", "")

            with open(filepath, "r") as f:
                content = json.load(f)

            value = content.get("objective_value") or content.get("cost")

            row = {
                "technique": technique,
                "run": content.get("run"),
                "elapsed_time_sec": content.get("elapsed_time_sec"),
                "value": value
            }

            data.append(row)

    return pd.DataFrame(data)

# Exemplo de uso
directory = "outputs/202301"
df = extract_data_from_jsons(directory)
print(df)
