"""
prepare_pb_data.py
Prepara cenarios multi-nivel para UF=PB:
  estoque (UF=PB) -> loja (revendedora) -> produto

Saida: data/pb_cenarios.csv
  Colunas: uf, loja, produto, venda_ciclo, demanda
  + metadata: data/pb_cenarios_meta.csv
  Colunas: uf, loja, produto, n_periodos, dem_total, dem_media, dem_std, cv
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MIN_PERIODOS = 17     # mínimo de ciclos para incluir uma combinação loja×produto
MAX_LOJAS    = 15     # máximo de lojas por produto (top por periodos)
PRODUTO_ALVO = 48130  # item com mais lojas cobertas em PB

def main():
    print("[1/4] Carregando dados...")
    df = pd.read_csv("data/vendas_sample.csv")

    # Filtro de qualidade: remove transacoes com valores impossiveis
    df = df[(df["sales"] >= 0) & (df["sales"] < 50_000)].copy()

    pb = df[df["warehouse"] == "PB"].copy()
    print(f"  PB limpo: {len(pb):,} registros | "
          f"{pb['store_id'].nunique()} lojas | "
          f"{pb['item_id'].nunique()} produtos | "
          f"{pb['venda_ciclo'].nunique()} ciclos")

    all_ciclos = sorted(pb["venda_ciclo"].unique())

    print(f"\n[2/4] Selecionando lojas para produto {PRODUTO_ALVO}...")
    item_pb = pb[pb["item_id"] == PRODUTO_ALVO]

    # Conta periodos por loja
    store_counts = item_pb.groupby("store_id")["venda_ciclo"].nunique()
    lojas_ok = store_counts[store_counts >= MIN_PERIODOS].sort_values(ascending=False)
    lojas_sel = lojas_ok.index[:MAX_LOJAS].tolist()

    print(f"  Lojas com >= {MIN_PERIODOS} ciclos: {len(lojas_ok)}")
    print(f"  Selecionadas (top {MAX_LOJAS}): {lojas_sel}")

    print(f"\n[3/4] Construindo series temporais completas...")
    cenarios = []
    meta = []

    for loja in lojas_sel:
        serie = (item_pb[item_pb["store_id"] == loja]
                 .groupby("venda_ciclo")["sales"].sum()
                 .reindex(all_ciclos, fill_value=0)
                 .reset_index())
        serie.columns = ["venda_ciclo", "demanda"]
        serie["uf"]     = "PB"
        serie["loja"]   = loja
        serie["produto"] = PRODUTO_ALVO
        cenarios.append(serie[["uf","loja","produto","venda_ciclo","demanda"]])

        dem = serie["demanda"].values
        meta.append({
            "uf":          "PB",
            "loja":        loja,
            "produto":     PRODUTO_ALVO,
            "n_periodos":  int((dem > 0).sum()),
            "dem_total":   float(dem.sum()),
            "dem_media":   float(dem[dem > 0].mean()) if (dem > 0).any() else 0.0,
            "dem_std":     float(dem[dem > 0].std())  if (dem > 0).sum() > 1 else 0.0,
            "cv":          (float(dem.std() / dem.mean()) if dem.mean() > 0 else 0.0),
        })

    df_cen  = pd.concat(cenarios, ignore_index=True)
    df_meta = pd.DataFrame(meta).sort_values("dem_media", ascending=False)

    print("\n[4/4] Salvando...")
    df_cen.to_csv("data/pb_cenarios.csv", index=False)
    df_meta.to_csv("data/pb_cenarios_meta.csv", index=False)

    print(f"\n=== Cenarios preparados ===")
    print(f"  Arquivo      : data/pb_cenarios.csv  ({len(df_cen):,} linhas)")
    print(f"  Meta         : data/pb_cenarios_meta.csv  ({len(df_meta)} lojas)")
    print(f"  Produto      : {PRODUTO_ALVO} | UF: PB")
    print(f"  Ciclos       : {len(all_ciclos)} (2023/01 a 2025/04)")
    print()
    print(df_meta.round(1).to_string(index=False))

if __name__ == "__main__":
    main()
