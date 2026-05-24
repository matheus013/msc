"""
pipeline_registry.py — Registra todos os pipelines do projeto.

Uso:
  kedro run                                    # pipeline completo
  kedro run --pipeline data_ingestion          # só ingestão
  kedro run --pipeline inventory_simulation    # só simulação
  kedro run --pipeline statistical_validation  # só testes estatísticos
  kedro run --pipeline reporting               # só relatórios
"""
from kedro.pipeline import Pipeline

from simulation.pipelines import (
    data_ingestion,
    demand_forecasting,
    inventory_simulation,
    statistical_validation,
    reporting,
    demand_profiling,
    policy_selection,
)


def register_pipelines() -> dict[str, Pipeline]:
    di  = data_ingestion.create_pipeline()
    df  = demand_forecasting.create_pipeline()
    inv = inventory_simulation.create_pipeline()
    sv  = statistical_validation.create_pipeline()
    rep = reporting.create_pipeline()
    dp  = demand_profiling.create_pipeline()
    ps  = policy_selection.create_pipeline()

    return {
        # Pipeline completo AIPE: ingestão → perfil → simulação → seleção → validação → relatório
        "__default__": di + df + dp + inv + sv + ps + rep,
        "data_ingestion":         di,
        "demand_forecasting":     df,
        "inventory_simulation":   inv,
        "statistical_validation": sv,
        "reporting":              rep,
        "demand_profiling":       dp,
        "policy_selection":       ps,
        # Atalhos compostos
        "data":             di + df,
        "simulation":       di + inv,
        "analysis":         sv + rep,
        "full_no_forecast": di + dp + inv + sv + ps + rep,
        # AIPE completo sem relatório (mais rápido para iteração)
        "aipe":             di + dp + inv + ps,
        # Só o engine de seleção (quando kpis e demand_features já existem)
        "policy_engine":    dp + ps,
    }
