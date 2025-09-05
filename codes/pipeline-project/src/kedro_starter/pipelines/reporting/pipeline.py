from __future__ import annotations
from kedro.pipeline import Pipeline, node

from kedro_starter.pipelines.data_enrich.nodes import enrich_venda


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=enrich_venda,
            inputs=["vendas_clean", "revendedor_clean", "produto_clean"],
            outputs="vendas_enrich",
            name="enrich_venda",
            tags=["enrich"],
        )
    ])
