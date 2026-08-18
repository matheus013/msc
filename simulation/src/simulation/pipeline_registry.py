"""
pipeline_registry.py — Registra todos os pipelines do projeto.

Uso:
  kedro run                                    # pipeline completo
  kedro run --pipeline data_ingestion          # só ingestão
  kedro run --pipeline inventory_simulation    # só simulação
  kedro run --pipeline statistical_validation  # só testes estatísticos
  kedro run --pipeline reporting               # só relatórios

Versão final da dissertação:
  kedro run --pipeline benchmark_final         # Experimento 2 (BA) reexecutado
  kedro run --pipeline benchmark_m5 --env m5   # comparação externa no Walmart M5
  kedro run --pipeline final_report            # só as análises e tabelas LaTeX
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
    final_report,
)


def register_pipelines() -> dict[str, Pipeline]:
    di  = data_ingestion.create_pipeline()
    df  = demand_forecasting.create_pipeline()
    inv = inventory_simulation.create_pipeline()
    sv  = statistical_validation.create_pipeline()
    rep = reporting.create_pipeline()
    dp  = demand_profiling.create_pipeline()
    ps  = policy_selection.create_pipeline()
    fr  = final_report.create_pipeline()
    dir_ = data_ingestion.create_resume_pipeline()
    # 2026-08-18: saida do AIPE pedida pelo usuario -- tabela com o score de
    # cada uma das 18 politicas por perfil operacional. Inclui `dp`
    # (classify_operational_profiles) de proposito: a criacao dos perfis
    # faz parte do processo, nao so o consumo de um demand_profiles.parquet
    # que ja exista. So depende de `kpis`/`scenarios`/`scenarios_meta`, que
    # benchmark_final/benchmark_bot/benchmark_m5 ja produzem -- roda DEPOIS,
    # sem reexecutar a simulacao (~18 politicas x N series e a parte cara).
    #
    # `ps` (policy_selection) incluido a pedido do usuario: o POD de `dp`
    # rotula a "politica dominante" por uma heuristica FIXA no codigo (nao
    # aprendida); `ps` e o classificador supervisionado de verdade --
    # aprende de kpis (serie x politica, o mesmo dado que gera o score da
    # tabela acima) qual politica recomendar por serie, via XGBoost treinado
    # em demand_features. Datasets isolados por ambiente em
    # conf/{bot,m5}/catalog.yml (senao reproduziria o incidente #2).
    prof_analysis = dp + reporting.create_profile_analysis_pipeline() + ps

    return {
        # Pipeline completo AIPE: ingestão → perfil → simulação → seleção → validação → relatório
        "__default__": di + df + dp + inv + sv + ps + rep,
        "data_ingestion":         di,
        "data_resume":            dir_,   # parte de sales_raw ja materializado
        "final_report":           fr,
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

        # ── Versão final da dissertação ──────────────────────────────────
        # Reexecução do Experimento 2 (Bahia) a partir de sales_raw, com o
        # portfólio ampliado e o código corrigido, seguida das análises e
        # tabelas LaTeX:
        #     kedro run --pipeline benchmark_final
        "benchmark_final":  dir_ + dp + inv + sv + fr,
        # Benchmark sobre a base pública Walmart M5, para comparação externa
        # com a literatura (Zabraoui et al., 2025). Usa o ambiente m5:
        #     kedro run --pipeline benchmark_m5 --env m5
        "benchmark_m5":     di + dp + inv + fr,
        # Benchmark sobre a base interna COMPLETA (todos os 27 estados,
        # apelido "bot" — em oposição ao recorte oficial da Bahia e à base
        # externa "m5"). Usa o ambiente bot (states=["all"], catálogo
        # isolado em data/*/bot/, pré-passo DuckDB para caber na RAM):
        #     kedro run --pipeline benchmark_bot --env bot
        "benchmark_bot":    di + dp + inv + fr,
        # Saida do AIPE: score de cada uma das 18 politicas por perfil
        # operacional (nao so a vencedora), recalculando os perfis como
        # parte do processo, MAIS o treino do PSE (classificador
        # supervisionado real, nao a heuristica fixa do POD). Roda depois de
        # benchmark_final/_bot/_m5 (usa os `kpis`/`scenarios` ja produzidos,
        # nao reexecuta a simulacao):
        #     kedro run --pipeline profile_analysis [--env bot|m5]
        "profile_analysis": prof_analysis,
    }
