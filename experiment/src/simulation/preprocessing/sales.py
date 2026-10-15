"""Conversor unico de vendas Parquet, com agregacao em disco via DuckDB."""
import argparse
from pathlib import Path

from simulation.paths import RAW_DATA_ROOT, ARTIFACTS_ROOT


def prepare_sales(source, output, states=None, cycles_per_year=17, overwrite=False):
    """Agrega particoes sem carregar a base inteira em pandas.

    Preserva warehouse por UF e combina chaves repetidas entre arquivos.
    Venda_ciclo e um ciclo comercial (YYYYCC), nao um mes calendario.
    """
    import duckdb

    source, output = Path(source).resolve(), Path(output).resolve()
    if cycles_per_year < 1:
        raise ValueError("cycles_per_year deve ser positivo")
    if output == source or source in output.parents:
        raise ValueError("A saida deve ficar fora da pasta de dados brutos")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Saida ja existe: {output}. Use --overwrite para substituir.")
    if not any(source.rglob("*.parquet")):
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = [str(state).upper() for state in (states or [])]
    query = """
        WITH raw AS (
            SELECT upper(CAST(uf AS VARCHAR)) AS warehouse,
                   CAST(produto_cod AS VARCHAR) AS item_id,
                   CAST(revendedor_cod AS VARCHAR) AS store_id,
                   CAST(venda_ciclo AS INTEGER) AS venda_ciclo,
                   CAST(venda_qtd AS DOUBLE) AS sales,
                   CAST(venda_vlr_receita_liquida AS DOUBLE) AS revenue
            FROM read_parquet(?, hive_partitioning=true, union_by_name=true)
            WHERE uf IS NOT NULL
        ), grouped AS (
            SELECT warehouse, item_id, store_id, venda_ciclo,
                   sum(sales) AS sales, sum(revenue) AS revenue
            FROM raw
            WHERE (? OR warehouse IN (SELECT unnest(?)))
            GROUP BY ALL
        )
        SELECT strftime(make_date(venda_ciclo // 100, 1, 1) +
                        CAST((venda_ciclo % 100 - 1) * ? AS INTEGER), '%Y-%m-%d') AS date,
               venda_ciclo, warehouse, item_id, store_id,
               sales, revenue, coalesce(revenue / nullif(sales, 0), 0) AS avg_price
        FROM grouped
        ORDER BY warehouse, item_id, store_id, venda_ciclo
    """
    with duckdb.connect() as connection:
        spill = ARTIFACTS_ROOT / "temporary" / "duckdb"
        spill.mkdir(parents=True, exist_ok=True)
        connection.execute("SET temp_directory = ?", [str(spill)])
        relation = connection.sql(query, params=[
            str(source / "**" / "*.parquet"), not selected, selected or [""],
            365 // cycles_per_year,
        ])
        relation.write_csv(str(output), header=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATA_ROOT / "source/vendas")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--states", nargs="+", help="Ex.: PB ou BA PB; omitir para todos")
    parser.add_argument("--cycles-per-year", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    output = options.output or ARTIFACTS_ROOT / "legacy" / (
        "vendas_sample.csv" if options.states == ["PB"] else "vendas_preparadas.csv")
    result = prepare_sales(options.source, output, options.states,
                           options.cycles_per_year, options.overwrite)
    print(f"Vendas preparadas: {result}")
    return 0
