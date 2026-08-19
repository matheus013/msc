# Validação — CTI Ajustado por Estabilidade Operacional

Gerado em: 2026-08-19 04:56

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
  High Vol. Seasonal: CTI_puro=VendorResponsive, CTI_adj(λ=0)=VendorResponsive — OK
  Sparse High Impact: CTI_puro=VendorResponsive, CTI_adj(λ=0)=VendorResponsive — OK
  Unstable Trend: CTI_puro=FixedInterval, CTI_adj(λ=0)=FixedInterval — OK

## Checagem 5 — Normalização
  High Vol. Seasonal: CTI_adj_selected(λ=0.25)=0.0588 ∈ [0,2] ✓
  Sparse High Impact: CTI_adj_selected(λ=0.25)=0.0659 ∈ [0,2] ✓
  Unstable Trend: CTI_adj_selected(λ=0.25)=0.0706 ∈ [0,2] ✓

## Checagem 6 — NS médio nas estratégias
  λ=0.0:
    A: política única (CTI puro): NS=0.774
    B: política única (CTI ajustado): NS=0.774
    C: seleção por perfil (CTI puro): NS=0.782
    D: seleção por perfil (CTI ajustado): NS=0.782
    E: oráculo por série (exploratório): NS=nan
  λ=0.25:
    A: política única (CTI puro): NS=0.774
    B: política única (CTI ajustado): NS=0.774
    C: seleção por perfil (CTI puro): NS=0.782
    D: seleção por perfil (CTI ajustado): NS=0.774
    E: oráculo por série (exploratório): NS=nan
  λ=0.5:
    A: política única (CTI puro): NS=0.774
    B: política única (CTI ajustado): NS=0.774
    C: seleção por perfil (CTI puro): NS=0.782
    D: seleção por perfil (CTI ajustado): NS=0.774
    E: oráculo por série (exploratório): NS=nan
  λ=1.0:
    A: política única (CTI puro): NS=0.774
    B: política única (CTI ajustado): NS=0.774
    C: seleção por perfil (CTI puro): NS=0.782
    D: seleção por perfil (CTI ajustado): NS=0.774
    E: oráculo por série (exploratório): NS=nan

## Checagem 7 — Consistência com Tabela 5.2
  EOQ CTI_mean (kpis atual): 628.42  (Tabela 5.2: 628.42 — rodada final)

## Limitações
- kpis.parquet é rodada anterior; valores serão atualizados após re-simulação.
- Perfis com n < 20: evidência exploratória.
- Oráculo por série: limite exploratório, não estratégia operacional.
- Análise concentrada no regime Lumpy do Experimento 2.