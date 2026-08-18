"""
m5_loader.py — Carga da base Walmart M5 no formato interno do pipeline.

Permite executar todo o AIPE sobre a base pública usada por Zabraoui et al.
(Supply Chain Analytics 12:100154, 2025), o artigo-base da arquitetura GA-DQN,
sem alterar nenhum estágio a jusante: a saída desta função tem exatamente o
mesmo schema do parquet transacional interno, de modo que os nós de filtragem,
limpeza e construção de cenários seguem inalterados.

Mapeamento de entidades
-----------------------
    warehouse  <- state_id   (CA, TX, WI)
    store_id   <- store_id   (CA_1, CA_2, ..., WI_3)
    item_id    <- item_id    (HOBBIES_1_001, ...)
    segmento   <- cat_id     (HOBBIES, HOUSEHOLD, FOODS)
    praca      <- state_id
    filial     <- store_id

Agregação temporal
------------------
O M5 é diário; o pipeline opera em ciclo comercial de ~21 dias. Os dias são
agrupados em blocos consecutivos de `days_per_cycle` e rotulados no formato
YYYYCC do calendário interno, preservando a semântica de `venda_ciclo`.

Essa agregação é o que torna as duas bases comparáveis. Sem ela a comparação é
inválida: o M5 diário tem CV² tipicamente < 1, enquanto o recorte brasileiro
por ciclo comercial está em 3,8-4,1. Agregar em janelas equivalentes é
condição necessária para que o benchmark meça diferença de política, e não
diferença de granularidade.

Custo de memória
----------------
`sales_train_evaluation.csv` tem 30.490 séries x 1.941 dias (116 MB) e
`sell_prices.csv` tem 226 MB. A leitura é feita em formato largo e derretida
apenas para as séries selecionadas, e o preço só é lido quando
`with_revenue: true`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"


def _cycle_labels(n_days: int, days_per_cycle: int, cycles_per_year: int,
                  start_year: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rótulos YYYYCC para cada dia, no mesmo formato de `venda_ciclo` interno.

    Retorna (índice_de_ciclo_por_dia, rótulo_por_ciclo).
    """
    cycle_idx = np.arange(n_days) // days_per_cycle
    n_cycles = int(cycle_idx.max()) + 1
    labels = []
    for c in range(n_cycles):
        year = start_year + c // cycles_per_year
        within = c % cycles_per_year + 1
        labels.append(f"{year}{within:02d}")
    return cycle_idx, np.asarray(labels)


def load_m5_as_internal(params: dict) -> pd.DataFrame:
    """
    Lê a base M5 e devolve DataFrame long no schema transacional interno:
        [warehouse, store_id, item_id, venda_ciclo, demand, revenue?,
         segmento, praca, filial, status, venda_tipo]

    As colunas `status` e `venda_tipo` são preenchidas com valores neutros
    para que os filtros de qualidade do pipeline interno não descartem tudo.
    """
    m5 = params.get("m5", {}) or {}
    root = Path(m5.get("path", "data/01_raw/m5_walmart"))
    if not root.is_absolute():
        root = Path.cwd() / root
    if not (root / SALES_FILE).exists():
        raise FileNotFoundError(
            f"Base M5 nao encontrada em {root}. Esperado: {SALES_FILE}. "
            "Baixe de https://github.com/Nixtla/m5-forecasts (datasets/m5.zip)."
        )

    days_per_cycle = int(m5.get("days_per_cycle", 21))
    cycles_per_year = int(params.get("cycles_per_year", 17))
    start_year = int(m5.get("start_year", 2011))
    states = m5.get("states")
    categories = m5.get("categories")
    max_series = m5.get("max_series")
    max_cycles = m5.get("max_cycles")
    with_revenue = bool(m5.get("with_revenue", False))
    seed = int(m5.get("seed", 42))

    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # ── cabecalho apenas: descobre colunas de dia sem ler nenhuma linha ────
    header_cols = pd.read_csv(root / SALES_FILE, nrows=0).columns.tolist()
    day_cols = [c for c in header_cols if c.startswith("d_")]
    day_cols.sort(key=lambda c: int(c.split("_")[1]))

    # Se max_cycles restringe o horizonte, ja descarta as colunas de dia
    # fora dele ANTES de ler — evita ler ~1940 colunas quando so ~800 (38
    # ciclos x 21 dias) sao usadas.
    if max_cycles:
        day_cols = day_cols[: int(max_cycles) * days_per_cycle]

    n_days = len(day_cols)
    cycle_idx, labels = _cycle_labels(n_days, days_per_cycle, cycles_per_year, start_year)
    n_cycles = int(cycle_idx.max()) + 1

    # ── leitura + filtro + agregacao em ciclos via DuckDB ──────────────────
    # sales_train_evaluation.csv: 30.490 series x 1.941 dias (116MB em
    # disco). Ler isso inteiro com pandas antes de filtrar/agregar mede
    # varios GB em RAM (overhead de parsing por coluna). DuckDB projeta so
    # as colunas declaradas em `columns=` (nao aloca as ~1100+ que sobram
    # quando max_cycles restringe o horizonte), aplica o WHERE de
    # states/categories durante a leitura (nao depois) e resolve a soma por
    # ciclo — que e uma expressao aritmetica entre colunas da MESMA linha,
    # nao um agregado entre linhas — em streaming, sem materializar a
    # tabela larga em nenhum momento. So cruza para pandas o resultado ja
    # agregado: no maximo 30.490 linhas x (5 + n_cycles) colunas.
    where = []
    if states:
        vals = ", ".join("'" + s.upper().replace("'", "''") + "'" for s in states)
        where.append(f"upper(state_id) IN ({vals})")
    if categories:
        vals = ", ".join("'" + str(c).replace("'", "''") + "'" for c in categories)
        where.append(f"cat_id IN ({vals})")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    cycle_exprs = []
    for c in range(n_cycles):
        cols_in_cycle = [day_cols[i] for i in range(n_days) if cycle_idx[i] == c]
        summed = " + ".join(f'"{col}"' for col in cols_in_cycle)
        cycle_exprs.append(f'({summed}) AS cycle_{c}')

    # `columns=` explicito (so id_cols+day_cols) faria o DuckDB tentar
    # decodificar o arquivo como se so tivesse essas colunas, e o sniffer
    # de dialeto do CSV quebra (conta 1946 colunas reais x 803 declaradas).
    # Deixa o auto-detect ler o cabecalho inteiro; a poda de colunas fica
    # a cargo do projection pushdown do otimizador na SELECT abaixo — o
    # ganho de memoria real vem de nunca materializar a tabela larga em
    # pandas, nao de poupar leitura de bytes do CSV.
    csv_path = (root / SALES_FILE).as_posix()
    query = f"""
        SELECT {", ".join(id_cols)}, {", ".join(cycle_exprs)}
        FROM read_csv('{csv_path}', header=true)
        {where_sql}
    """
    log.info("M5: lendo %s via DuckDB (states=%s categories=%s, %d colunas de dia agregadas em %d ciclos)",
              csv_path, states, categories, n_days, n_cycles)
    con = duckdb.connect()
    import os as _os
    con.execute(f"PRAGMA threads={max(1, (_os.cpu_count() or 4) - 2)}")
    sales = con.execute(query).df()
    con.close()
    if sales.empty:
        raise ValueError(f"Nenhuma serie M5 apos filtros states={states} categories={categories}")
    log.info("M5: %d series apos filtro states=%s categories=%s (ja agregadas em %d ciclos)",
             len(sales), states, categories, n_cycles)

    agg = sales[[f"cycle_{c}" for c in range(n_cycles)]].to_numpy(dtype=np.float32)

    # ── suavização de outliers (1,5x IQR), se pedido ────────────────────
    # 2026-08-18: reproduz o tratamento de dados descrito na Seção 3.7 do
    # artigo-base ("Outliers -- especially during holiday demand surges --
    # were smoothed using a robust 1.5 x IQR filtering method to reduce
    # noise"). Aplicado POR SÉRIE (cada linha de `agg` é uma série
    # loja-produto ao longo dos ciclos): valores acima de Q3 + 1,5*IQR são
    # truncados nesse teto. Só a cauda superior importa aqui -- picos de
    # demanda em feriados, que é o caso citado no artigo; o piso natural da
    # demanda já é 0, então não há cauda inferior a truncar.
    if bool(m5.get("iqr_outlier_smoothing", True)):
        q1 = np.percentile(agg, 25, axis=1, keepdims=True)
        q3 = np.percentile(agg, 75, axis=1, keepdims=True)
        fence = q3 + 1.5 * (q3 - q1)
        n_clipped = int((agg > fence).sum())
        agg = np.minimum(agg, fence).astype(np.float32)
        log.info("M5: suavizacao 1,5xIQR aplicada -- %d valores truncados de %d",
                 n_clipped, agg.size)

    # ── amostra de séries, se pedido ─────────────────────────────────────
    meta = sales[id_cols].reset_index(drop=True)
    if max_series and len(meta) > int(max_series):
        rng = np.random.default_rng(seed)
        # prioriza séries com alguma demanda, depois amostra sem viés de volume
        nonzero = np.flatnonzero(agg.sum(axis=1) > 0)
        pick = rng.choice(nonzero, size=min(int(max_series), nonzero.size),
                          replace=False)
        pick.sort()
        meta = meta.iloc[pick].reset_index(drop=True)
        agg = agg[pick]
        log.info("M5: amostradas %d series (seed=%d)", len(meta), seed)

    # ── formato long ─────────────────────────────────────────────────────
    n_series, n_cyc = agg.shape
    df = pd.DataFrame({
        "warehouse": np.repeat(meta["state_id"].to_numpy(), n_cyc),
        "store_id":  np.repeat(meta["store_id"].to_numpy(),  n_cyc),
        "item_id":   np.repeat(meta["item_id"].to_numpy(),   n_cyc),
        "segmento":  np.repeat(meta["cat_id"].to_numpy(),    n_cyc),
        "venda_ciclo": np.tile(labels[:n_cyc], n_series),
        "demand":    agg.reshape(-1),
    })
    df["praca"] = df["warehouse"]
    df["filial"] = df["store_id"]
    # valores neutros para os filtros do pipeline interno
    df["status"] = "Ativo"
    df["venda_tipo"] = "Normal"

    # ── receita (opcional; exige ler sell_prices, 226 MB) ────────────────
    if with_revenue:
        df["revenue"] = _attach_revenue(df, root, days_per_cycle, labels,
                                        cycles_per_year, start_year)
    else:
        # receita proxy: mantém a coluna para os relatórios que a esperam
        df["revenue"] = df["demand"]

    log.info("M5 carregado: %d linhas | %d series | %d ciclos | estados=%s",
             len(df), n_series, n_cyc,
             sorted(df["warehouse"].unique().tolist()))
    return df


def _attach_revenue(df: pd.DataFrame, root: Path, days_per_cycle: int,
                    labels: np.ndarray, cycles_per_year: int,
                    start_year: int) -> pd.Series:
    """
    Receita = demanda do ciclo x preço médio do item-loja no ciclo.

    O M5 fornece preço semanal (`wm_yr_wk`); o calendário mapeia dia -> semana.
    Aproximamos o preço do ciclo pela média das semanas que ele cobre.
    """
    cal = pd.read_csv(root / CALENDAR_FILE, usecols=["d", "wm_yr_wk"])
    cal["day_n"] = cal["d"].str.split("_").str[1].astype(int)
    cal = cal.sort_values("day_n")
    cal["cycle"] = (cal["day_n"] - 1) // days_per_cycle
    cal = cal[cal["cycle"] < len(labels)]
    cal["venda_ciclo"] = labels[cal["cycle"].to_numpy()]
    wk2cycle = cal[["wm_yr_wk", "venda_ciclo"]].drop_duplicates()

    keys = df[["store_id", "item_id"]].drop_duplicates()
    prices = pd.read_csv(root / PRICES_FILE)
    prices = prices.merge(keys, on=["store_id", "item_id"], how="inner")
    prices = prices.merge(wk2cycle, on="wm_yr_wk", how="inner")
    price_cycle = (prices.groupby(["store_id", "item_id", "venda_ciclo"])
                   ["sell_price"].mean().reset_index())

    merged = df[["store_id", "item_id", "venda_ciclo", "demand"]].merge(
        price_cycle, on=["store_id", "item_id", "venda_ciclo"], how="left")
    price = merged["sell_price"].fillna(merged["sell_price"].median()).fillna(0.0)
    return (merged["demand"] * price).to_numpy()
