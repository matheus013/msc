from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.policy_selection.nodes import (
    generate_policy_labels,
    train_policy_selector,
    evaluate_profile_feature_gain,
    apply_policy_selector,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_policy_labels,
            inputs=["kpis", "demand_features", "params:policy_selection"],
            outputs="policy_labels",
            name="generate_policy_labels",
        ),
        # 2026-08-18: `train_policy_selector`/`apply_policy_selector` passaram
        # a receber `demand_profiles` (nao mais `demand_features`) -- o
        # perfil operacional entra como feature one-hot, a pedido do
        # usuario ("AIPE e selecao de politica por perfil, nao faz sentido
        # nao usar perfil como entrada"). Ver nodes.py.
        node(
            func=train_policy_selector,
            inputs=["demand_profiles", "policy_labels", "params:policy_selection"],
            outputs=["policy_selector_model", "policy_selector_metrics"],
            name="train_policy_selector",
        ),
        # Responde diretamente "o ganho de usar perfil e relevante?":
        # treina COM e SEM o perfil, mesmo split de CV, compara.
        node(
            func=evaluate_profile_feature_gain,
            inputs=["demand_profiles", "policy_labels", "params:policy_selection"],
            outputs="policy_selector_profile_gain",
            name="evaluate_profile_feature_gain",
        ),
        node(
            func=apply_policy_selector,
            inputs=["demand_profiles", "policy_selector_model"],
            outputs="policy_recommendations",
            name="apply_policy_selector",
        ),
    ])
