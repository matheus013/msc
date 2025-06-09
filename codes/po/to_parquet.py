import os
import modin.pandas as pd  # Apenas troca o pandas normal
from pathlib import Path

def detect_format(filename):
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".csv": "csv",
        ".json": "json",
        ".xlsx": "xlsx"
    }.get(ext, None)

def main():
    input_dir = "data/raw"
    output_dir = "data/raw"

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if not os.path.isfile(file_path):
            continue

        file_format = detect_format(file)
        if not file_format:
            print(f"❌ Ignorando {file}: formato não suportado")
            continue

        print(f"➡️ Processando: {file} como {file_format}")

        try:
            if file_format == "csv":
                df = pd.read_csv(file_path)
            elif file_format == "json":
                df = pd.read_json(file_path, lines=False)
            elif file_format == "xlsx":
                df = pd.read_excel(file_path, engine="openpyxl")
            else:
                continue

            output_file = os.path.join(output_dir, Path(file).stem + ".parquet")
            # Força colunas de tipo 'object' para string
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype("string")

            df.to_parquet(output_file, engine="pyarrow", index=False)
            print(f"✅ Arquivo salvo em: {output_file}")

        except Exception as e:
            print(f"❌ Erro ao processar {file}: {e}")

# 👇 ESSENCIAL NO WINDOWS
if __name__ == "__main__":
    main()
