"""
final_report/pipeline.py — Analises e tabelas da versao final.

Substitui os scripts avulsos que existiam na raiz do projeto. Consome `kpis`
produzido pelo pipeline de simulacao e entrega redundancia, estratificacao por
volume, confronto com os numeros da proposta e as tabelas LaTeX.

    kedro run --pipeline final_report
"""
from kedro.pipeline import Pipeline, node, pipeline

from simulation.pipelines.final_report.nodes import (
    build_final_latex_tables,
    compare_with_proposal,
    compute_policy_redundancy,
    stratify_by_volume,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_policy_redundancy,
            inputs=["kpis", "params:final_report"],
            outputs="policy_redundancy",
            name="compute_policy_redundancy",
        ),
        node(
            func=stratify_by_volume,
            inputs=["kpis", "params:final_report"],
            outputs="kpis_by_volume",
            name="stratify_by_volume",
        ),
        node(
            func=compare_with_proposal,
            inputs=["kpis", "params:final_report"],
            outputs="proposal_comparison",
            name="compare_with_proposal",
        ),
        node(
            func=build_final_latex_tables,
            inputs=["kpis", "policy_redundancy", "proposal_comparison",
                    "params:final_report"],
            outputs="final_latex_tables",
            name="build_final_latex_tables",
        ),
    ])
