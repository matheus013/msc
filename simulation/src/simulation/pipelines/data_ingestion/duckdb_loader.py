"""
duckdb_loader.py — pre-passo GLOBAL, via DuckDB, para a base interna
COMPLETA (todos os estados), apelidada de "bot" nas conversas do projeto
(em oposicao a "m5" / Walmart, a base publica externa).

Motivacao
---------
`data/01_raw/vendas/uf=XX/*.parquet` soma 55,3M linhas / ~1,9GB em 27
particoes. `filter_by_parameters()` calcula "produtos ativos" (janela de
ciclos recentes) e CV minimo GLOBALMENTE por `item_id`, agregando demanda
de TODOS os estados juntos — nao da pra decidir esses dois filtros
particao por particao sem mudar o resultado (um item pode ser "ativo" ou
ter CV suficiente so quando o pais inteiro entra na conta).

A tentativa obvia — rodar o filtro inteiro em DuckDB e materializar o
resultado em pandas — foi tentada e descartada: mesmo depois de todos os
filtros, sobraram 27M linhas (49% do bruto; o filtro de CV pooled entre
estados reduz muito menos que dentro de um unico estado, onde caiu pra
8,6% na Bahia) e o `.df()` final quase estourou a RAM da maquina (1,4GB
livres de 32GB, com a run da Bahia rodando em paralelo). Ver conversa de
2026-08-18 (incidente de sales_raw.parquet + quase-OOM deste teste).

Alem disso, `clean_sales_data()` deduplica por
(warehouse,item_id,store_id,venda_ciclo) mantendo a PRIMEIRA linha na
ordem de leitura — checado empiricamente na Bahia: ~4,5% dos grupos
duplicados tem `venda_qtd` DIFERENTE entre as copias, entao "qual
sobrevive" muda o valor de demanda agregado. Replicar isso em SQL exigiria
garantir a mesma ordem de varredura do pandas (particao por particao,
arquivo por arquivo) — arriscado o suficiente pra nao valer o atalho.

Estrategia adotada
-------------------
So a parte comprovadamente segura de paralelizar/agregar roda em DuckDB —
os filtros GLOBAIS de produtos ativos + CV (streaming, nunca materializa
as 55M linhas, so devolve a lista de `item_id` sobreviventes: algumas
centenas). Essa lista alimenta o parametro `products` do pipeline
(`filter_by_parameters` ja sabe filtrar por produto especifico), e
`load_raw_sales()` volta a ler particao por particao com pandas — MESMO
CAMINHO ja usado e validado para um unico estado — mas agora cada
particao ja e filtrada por (status, venda_tipo, produtos validos) ANTES
de concatenar, entao o pico de memoria fica limitado ao tamanho de UMA
particao (a maior, CE, tem 8,3M linhas / 186MB), nao a soma das 27.

Dedup, periodo e CV continuam rodando de novo em `filter_by_parameters` /
`clean_sales_data` sem nenhuma mudanca de codigo — sao os MESMOS nos,
sobre dados MENORES.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


def _sql_list(vals):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in vals)


def compute_valid_products(params: dict, root: Path) -> list[str]:
    """
    Replica, em SQL, EXATAMENTE os filtros globais/cross-estado de
    `filter_by_parameters()` — status, venda_tipo, produtos ativos
    (janela de ciclos), periodo, segmento, CV minimo, NESSA ORDEM, com os
    MESMOS limiares de `params:data_ingestion` — e devolve so a lista de
    `produto_cod` que sobrevive a todos eles.

    Streaming: nunca materializa as linhas em pandas, so contagens e a
    lista final de produtos (algumas centenas). Dedup e o filtro de
    periodo por linha continuam acontecendo depois, em pandas, sobre os
    dados ja restritos a essa lista — esta funcao so decide QUAIS
    produtos entram, nao filtra linha a linha.
    """
    glob = str(root / "uf=??" / "*.parquet")

    active_statuses = params.get("active_statuses", ["Ativo"])
    exclude_tipos = params.get("exclude_venda_tipos") or []
    window = int(params.get("active_product_window", 3))
    date_start = params.get("date_start")
    date_end = params.get("date_end")
    segmentos = params.get("segmentos")
    cv_threshold = float(params.get("cv_threshold", 0.0))

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 4) - 2)}")

    where = []
    if active_statuses:
        where.append(f"status IN ({_sql_list(active_statuses)})")
    if exclude_tipos:
        where.append(f"venda_tipo NOT IN ({_sql_list(exclude_tipos)})")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    con.execute(f"""
        CREATE TEMP TABLE t1 AS
        SELECT produto_cod, venda_ciclo, segmento,
               COALESCE(TRY_CAST(venda_qtd AS DOUBLE), 0) AS demand
        FROM read_parquet('{glob}', hive_partitioning=true)
        {where_sql}
    """)
    n1 = con.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
    log.info("[bot/duckdb pre-passo] status=%s venda_tipo(excl)=%s: %d linhas (streaming)",
             active_statuses, exclude_tipos, n1)

    if window > 0:
        con.execute(f"""
            CREATE TEMP TABLE recent_cycles AS
            SELECT DISTINCT venda_ciclo FROM t1 ORDER BY venda_ciclo DESC LIMIT {window}
        """)
        con.execute("""
            CREATE TEMP TABLE t2 AS
            SELECT * FROM t1 WHERE produto_cod IN (
                SELECT DISTINCT produto_cod FROM t1
                WHERE venda_ciclo IN (SELECT venda_ciclo FROM recent_cycles)
            )
        """)
    else:
        con.execute("CREATE TEMP TABLE t2 AS SELECT * FROM t1")
    n_items2 = con.execute("SELECT COUNT(DISTINCT produto_cod) FROM t2").fetchone()[0]
    log.info("[bot/duckdb pre-passo] produtos ativos (janela=%d ciclos, global): %d produtos",
             window, n_items2)

    date_where = []
    if date_start:
        date_where.append(f"venda_ciclo >= '{date_start}'")
    if date_end:
        date_where.append(f"venda_ciclo <= '{date_end}'")
    date_sql = ("WHERE " + " AND ".join(date_where)) if date_where else ""
    con.execute(f"CREATE TEMP TABLE t3 AS SELECT * FROM t2 {date_sql}")

    if segmentos:
        con.execute(f"CREATE TEMP TABLE t4 AS SELECT * FROM t3 WHERE segmento IN ({_sql_list(segmentos)})")
    else:
        con.execute("CREATE TEMP TABLE t4 AS SELECT * FROM t3")

    if cv_threshold > 0:
        rows = con.execute(f"""
            SELECT produto_cod
            FROM (
                SELECT produto_cod,
                       STDDEV_SAMP(demand) / (AVG(demand) + 1e-9) AS cv
                FROM t4
                GROUP BY produto_cod
            )
            WHERE cv >= {cv_threshold}
        """).fetchall()
    else:
        rows = con.execute("SELECT DISTINCT produto_cod FROM t4").fetchall()
    con.close()

    valid = [r[0] for r in rows]
    log.info("[bot/duckdb pre-passo] Filtro CV >= %.2f (global): %d produtos validos "
             "(essa lista alimenta filter_by_parameters.products a jusante)",
             cv_threshold, len(valid))
    return valid
