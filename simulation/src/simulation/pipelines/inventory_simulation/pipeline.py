from kedro.pipeline import Pipeline, node, pipeline
from simulation.pipelines.inventory_simulation.nodes import (
    scale_parameters_per_store,
    run_classical_policies,
    run_metaheuristic_policies,
    run_rl_policies,
    run_proposed_architecture,
    aggregate_kpis,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scale_parameters_per_store,
            inputs=["scenarios_meta", "params:simulation"],
            outputs="scaled_params",
            name="scale_parameters_per_store",
        ),
        node(
            func=run_classical_policies,
            inputs=["scenarios", "scenarios_meta", "scaled_params", "params:simulation"],
            outputs="kpis_classical",
            name="run_classical_policies",
        ),
        node(
            func=run_metaheuristic_policies,
            inputs=["scenarios", "scenarios_meta", "scaled_params", "params:simulation"],
            outputs="kpis_metaheuristic",
            name="run_metaheuristic_policies",
        ),
        node(
            func=run_rl_policies,
            inputs=["scenarios", "scenarios_meta", "scaled_params", "params:simulation"],
            outputs="kpis_rl",
            name="run_rl_policies",
        ),
        node(
            func=run_proposed_architecture,
            inputs=["scenarios", "scenarios_meta", "scaled_params", "params:simulation"],
            outputs="kpis_proposed",
            name="run_proposed_architecture",
        ),
        node(
            func=aggregate_kpis,
            inputs=["kpis_classical", "kpis_metaheuristic", "kpis_rl", "kpis_proposed"],
            outputs="kpis",
            name="aggregate_kpis",
        ),
    ])
