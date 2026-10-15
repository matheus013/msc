"""Literature-supported candidate portfolio, reviewed on 2026-10-14.

This is an a-priori family/interpretability choice, not a statistical claim that
retired policies are equivalent or dominated. Historical matrices keep all arms.
"""
ACTIVE_POLICIES = (
    'EOQCycle', 'sSCycle', 'CappedBaseStock', 'GA', 'PPO', 'GA-DQN',
)
PORTFOLIO_STATUS = 'candidate_pending_correctness_and_runtime_validation'
HYBRID_ABLATION_POLICIES = ('GA', 'DQN', 'GA-DQN')

RETIRED_FROM_PRIMARY = {
    'FixedInterval':'Omit Zabraoui-associated comparison arm; retain capped base-stock as the primary order-up-to reference.',
    'NewsvendorHeuristic':'Single-period-inspired heuristic is auxiliary for this multi-period, positive-lead-time model.',
    'BigDataNewsvendor':'Contextual single-period extension is auxiliary; not a faithful replication of Ban and Rudin.',
    'VendorResponsive':'Omit the paper-specific adaptive formulation and the Zabraoui comparison family from primary scope.',
    'SA':'Retain GA as the representative threshold-search optimizer.',
    'DQN':'Required matched ablation for GA-DQN; not a redundant algorithm or a deleted implementation.',
}
