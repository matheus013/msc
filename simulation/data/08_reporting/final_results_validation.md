# Validação Final — Política Única Global vs. Seleção por Perfil (Fase 2, BA)

Gerado em: 2026-06-21

## 1. Arquivos usados

- `data/07_model_output/kpis.parquet` — KPIs por (série, política), agregados sobre 5 replicações
- `data/04_feature/demand_profiles.parquet` — Perfil Operacional de Demanda (POD) por série
- Scripts: `simulation/src/reporting/profile_policy_analysis.py`, `strategy_cost_comparison.py`, `cti_adjusted_analysis.py`
- Saídas: `simulation/data/08_reporting/profiles/`, `simulation/data/08_reporting/strategy/`

## 2. Número de séries

145 séries (loja, produto) — estado da Bahia, Fase 2, regime *Lumpy*. Confirmado em todas as checagens automáticas dos 3 scripts (`Esperado: 145 | Encontrado: 145`).

## 3. Número de políticas

12 políticas avaliadas, idênticas à Tabela 5.2 (`tab:kpis_fase2`): EOQ, (s,S), Jornaleiro (Newsvendor), GA, SA, PSO, DE, DQN, PPO, SARSA, GA-DQN, GA-PPO.

## 4. Regra da política única global (A1/A2)

A1 = política com menor CTI médio entre as politicamente viáveis (NS médio global ≥ 0,70). Viáveis: DE, EOQ, GA, GA-DQN, GA-PPO, PPO, PSO, SA, sS. A1 = **EOQ** (CTI médio = R$ 628,42; NS médio = 0,942).
A2 = EOQ adotado diretamente como *baseline* operacional de mercado. Neste experimento, A1 = A2 = EOQ.

## 5. Regra da seleção por perfil (B)

Para cada Perfil Operacional de Demanda (POD), seleciona-se a política de menor CTI médio entre as politicamente viáveis no perfil (NS médio do perfil ≥ 0,70); se nenhuma política for viável, usa-se *fallback* por maior NS médio (não ocorreu nos 3 perfis presentes na Fase 2).

| Perfil | n séries | Política dominante | CTI médio (R$) | NS médio | Observação |
|---|---|---|---|---|---|
| Sparse High Impact | 116 | **SA** | 584,01 | 0,865 | n ≥ 20 |
| Unstable Trend | 18 | **EOQ** | 607,95 | 0,947 | n < 20 — exploratório |
| High Vol. Seasonal | 11 | **EOQ** | 618,79 | 0,970 | n < 20 — exploratório |

Nota importante: o Jornaleiro (Newsvendor) tem o menor CTI absoluto no perfil Sparse High Impact (R$ 283,67), mas seu NS médio nesse perfil é 0,551 — **abaixo do limiar de viabilidade de 0,70** — e por isso é excluído da dominância, mesmo sendo o de menor custo bruto. Da mesma forma, o SARSA (NS médio 0,611 em High Vol. Seasonal) não atinge o limiar e não é dominante nesse perfil.

## 6. Regra do CTI ajustado (preliminar/complementar)

`J(pi, g) = CTI_norm(pi, g) + lambda_BE * BE_norm(pi, g)`, com `CTI_norm` e `BE_norm` normalizados por min-max sobre as 12 políticas (escopo global). `lambda_BE = 0` recupera o CTI puro (verificado: Checagem 4 dos 3 perfis bate exatamente com a política dominante por CTI puro). Avaliado para lambda_BE em {0; 0,25; 0,5; 1,0}. Esta é uma análise de sensibilidade complementar e preliminar — não substitui o CTI original como critério principal.

## 7. Fórmula de redução percentual

CTI puro: `100 * (CTI_A1 - CTI_estrategia) / CTI_A1`
CTI ajustado: `100 * (CTI_ajustado_A1 - CTI_ajustado_estrategia) / CTI_ajustado_A1`

Ambas verificadas numericamente contra as tabelas geradas (`table_strategy_comparison.tex`, `table_strategy_adjusted.tex`).

## 8. Principais resultados (CTI puro)

| Estratégia | CTI total (R$) | CTI médio (R$) | NS médio | Red. vs A1 (%) |
|---|---|---|---|---|
| A1: política única (melhor viável = EOQ) | 91.120,69 | 628,42 | 0,942 | 0,0 |
| A2: política única (EOQ, *baseline*) | 91.120,69 | 628,42 | 0,942 | 0,0 |
| B: seleção por perfil | 85.495,48 | 589,62 | 0,883 | **+6,2** |
| C†: oráculo por série (exploratório) | 50.433,59 | 347,82 | 0,700 | +44,6 |

† Não é estratégia operacional; limite inferior exploratório (conhecimento perfeito da política ótima por série).

**Leitura**: a redução de CTI da estratégia B (seleção por perfil) é modesta (6,2%), mas vem acompanhada de queda real de NS (0,942 → 0,883, ainda acima do limiar de viabilidade de 0,70). Não é um ganho puro — é um *trade-off* custo-serviço, consistente com a alocação do SA (NS=0,865) no perfil que concentra 80% das séries. O oráculo (C) mostra um limite superior teórico de 44,6% de redução, mas com NS médio caindo ao piso de viabilidade (0,700), reforçando que ganhos de custo maiores exigiriam relaxar a restrição de serviço.

## 9. CTI ajustado (sensibilidade, lambda_BE > 0)

Com a penalidade por instabilidade ativada, a política única ótima migra de EOQ (BE=55,4, alto) para SA (BE=5,5) a partir de lambda_BE=0,25, produzindo redução de CTI ajustado de até +56,6% (lambda_BE=1,0) em relação à estratégia A com CTI puro — às custas de NS médio global cair de 0,942 para 0,862. A seleção por perfil (D) converge para o mesmo resultado de B em lambda_BE ≥ 0,5, isto é, perde sua vantagem diferenciadora quando a penalidade por instabilidade domina o critério, porque a política mais estável (SA) passa a ser preferida uniformemente em todos os perfis.

## 10. Limitações

- Análise restrita ao regime *Lumpy*, Fase 2 (Bahia, 145 séries); generalização para outros regimes é objetivo da Fase 3.
- Perfis Low Vol. Stable e Fast Moving não têm representação na Fase 2.
- Perfis com n < 20 séries (Unstable Trend, High Vol. Seasonal) são evidência exploratória, não conclusiva.
- O oráculo por série (C) é um limite superior teórico, não uma política implementável.
- O CTI ajustado é uma análise de sensibilidade complementar e preliminar; o critério principal de viabilidade/dominância da dissertação permanece o CTI puro sob a restrição NS ≥ 0,70.
