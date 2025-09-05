from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import filter_active_products, clean_products, clean_revendedor


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=clean_products,
            inputs="raw_produto",
            outputs="produto_clean",
            name="clean_products",
            tags=["clean"],
        ),
        node(
            func=clean_revendedor,
            inputs="raw_revendedor",
            outputs="revendedor_clean",
            name="clean_revendedor",
            tags=["clean"],
        ),
        node(
            func=filter_active_products,
            inputs=["raw_vendas", "produto_clean", "params:sample_frac", "params:random_seed"],
            outputs="vendas_clean",
            name="filter_active_products",
            tags=['clean']
        ),
    ])
