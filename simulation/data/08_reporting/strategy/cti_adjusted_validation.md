# Validação — CTI Ajustado por Estabilidade Operacional

Gerado em: 2026-06-26 09:31

## Fonte dos dados
- `data/07_model_output/kpis.parquet`
- `data/04_feature/demand_profiles.parquet`

## Checagem 1 — Séries
  Esperado: 145 | Encontrado: 145

## Checagem 2 — Políticas degeneradas marcadas
  DEGENERATE_POLICIES = ['DQN', 'PPO']

## Checagem 3 — CTI original preservado
  A coluna 'CTI' nos artefatos CSV reproduz o TIC original do kpis.parquet sem modificação.

## Checagem 4 — lambda=0 recupera CTI puro
  High Vol. Seasonal: CTI_puro=SARSA, CTI_adj(λ=0)=SARSA — OK
  Sparse High Impact: CTI_puro=Newsvendor, CTI_adj(λ=0)=Newsvendor — OK
  Unstable Trend: CTI_puro=EOQ, CTI_adj(λ=0)=EOQ — OK

## Checagem 5 — Normalização
  High Vol. Seasonal: CTI_adj_selected(λ=0.25)=0.0073 ∈ [0,2] ✓
  Sparse High Impact: CTI_adj_selected(λ=0.25)=0.0000 ∈ [0,2] ✓
  Unstable Trend: CTI_adj_selected(λ=0.25)=0.0914 ∈ [0,2] ✓

## Checagem 6 — NS médio nas estratégias
  λ=0.0:
    A: política única (CTI puro): NS=0.954
    B: política única (CTI ajustado): NS=0.954
    C: seleção por perfil (CTI puro): NS=0.742
    D: seleção por perfil (CTI ajustado): NS=0.742
    E: oráculo por série (exploratório): NS=nan
  λ=0.25:
    A: política única (CTI puro): NS=0.954
    B: política única (CTI ajustado): NS=0.954
    C: seleção por perfil (CTI puro): NS=0.742
    D: seleção por perfil (CTI ajustado): NS=0.737
    E: oráculo por série (exploratório): NS=nan
  λ=0.5:
    A: política única (CTI puro): NS=0.954
    B: política única (CTI ajustado): NS=0.954
    C: seleção por perfil (CTI puro): NS=0.742
    D: seleção por perfil (CTI ajustado): NS=0.737
    E: oráculo por série (exploratório): NS=nan
  λ=1.0:
    A: política única (CTI puro): NS=0.954
    B: política única (CTI ajustado): NS=0.954
    C: seleção por perfil (CTI puro): NS=0.742
    D: seleção por perfil (CTI ajustado): NS=0.707
    E: oráculo por série (exploratório): NS=nan

## Checagem 7 — Consistência com Tabela 5.2
  EOQ CTI_mean (kpis atual): 512.21  (Tabela 5.2: 628.42 — rodada final)

## Limitações
- kpis.parquet é rodada anterior; valores serão atualizados após re-simulação.
- Perfis com n < 20: evidência exploratória.
- Oráculo por série: limite exploratório, não estratégia operacional.
- Análise concentrada no regime Lumpy do Experimento 2.