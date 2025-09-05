from __future__ import annotations
from kedro.pipeline import Pipeline, node
from kedro_starter.pipelines.reporting.nodes import generate_sales_reports

def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=generate_sales_reports,
            inputs=["vendas_enrich", "params:reports_dir"],
            outputs=None,  # side-effects: saves files to disk
            name="generate_sales_reports",
            tags=["reporting"],
        )
    ])
