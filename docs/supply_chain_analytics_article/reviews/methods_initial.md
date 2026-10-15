# Auditoria inicial de métodos e evidências — 14/10/2026

**Decisão: revisão maior; os resultados atuais sustentam uma reanálise exploratória, não a validação prospectiva de um seletor adaptativo superior.** Nenhum experimento histórico foi sobrescrito. A análise nova usa somente a matriz preservada de KPIs.

## Resultado principal verificado

Fonte computável: `experiment/scripts/analysis/sca_validation.py`; saídas: `experiment/data/08_reporting/sca_article/`. `validation.json` contém SHA256 dos dois arquivos de entrada, população, comparações e qualificações dos intervalos. Execute com `experiment/.venv/Scripts/python.exe experiment/scripts/analysis/sca_validation.py`. A reanálise usa CPU porque a matriz é pequena e não contém treinamento; transferi-la para GPU não acelera o agrupamento e a reamostragem.

Há **145 séries loja-produto, 115 lojas e 12 políticas**, com cobertura completa 145×12=1.740 linhas. Os perfis são Sparse High Impact (116), Unstable Trend (18) e High Volatility Seasonal (11). Os metadados têm 38 ciclos, e a fonte de cenários tem 5.510 linhas (145×38), entre 202301 e 202504. Os denominadores das frequências e custos de manutenção são compatíveis com 21 ciclos de avaliação; o parâmetro atual é treino nos primeiros 17 e avaliação nos 21 restantes. Essa compatibilidade não substitui um manifesto histórico de execução.

| Estratégia, limiar médio 0,70 | Custo médio | Serviço médio | Séries abaixo de 0,70 |
|---|---:|---:|---:|
| EOQ | 628,4186 | 0,942191 | 2 |
| SA | 1.147,8619 | 0,864921 | 24 |
| Melhor global, VendorResponsive | 433,3051 | 0,773816 | 40 |
| Perfil, seleção na própria amostra | 432,4598 | 0,781758 | 38 |
| Perfil, leave-one-series-out | 435,8007 | 0,781608 | 39 |
| Perfil, leave-one-store-out | 435,8007 | 0,781608 | 39 |

A seleção retrospectiva escolhe VendorResponsive em Sparse High Impact e High Volatility Seasonal, FixedInterval em Unstable Trend. Na seleção cruzada, as escolhas são VendorResponsive para 130 séries e FixedInterval para 15. A melhor global não muda em nenhuma dobra a 0,70. As duas formas de exclusão dão o mesmo resultado nesse limiar; não são avaliações temporais.

- Vantagem de custo retrospectiva por perfil contra melhor global: **0,1951%**, IC95% bootstrap por loja **[-1,0059%; 1,4272%]**.
- Vantagem leave-one-store-out: **−0,5760%**, IC95% **[-1,4296%; 0,1866%]**. Portanto o custo aumenta ligeiramente, sem evidência de vantagem de custo; a diferença de serviço é +0,007791, IC95% [0,003382; 0,012774].
- Contra EOQ, o perfil leave-one-store-out reduz custo **30,6512%**, IC95% [26,2295%; 34,2931%], mas reduz serviço em **0,160583** (16,06 pontos percentuais), IC95% [-0,184653; -0,138665]. Trata-se de troca custo-serviço.
- Intervalos: 5.000 reamostragens de 115 lojas com reposição, mantendo juntas suas séries. Condicionam-se aos perfis, KPIs e seletores já ajustados; a seleção não é reajustada dentro do bootstrap. As dobras de treinamento se sobrepõem. Esses intervalos não cobrem incerteza de treinamento nem demonstram desempenho em novos períodos ou outra rede.
- Sensibilidade executada em 0,70; 0,80; 0,85; 0,90; 0,95. A melhor global muda de VendorResponsive para FixedInterval (0,80), EOQ (0,85–0,90), PPO (0,95). Não escolher um limiar a posteriori porque favorece o perfil.

## Achados impeditivos de alegações fortes

1. **Relatórios e artefatos têm versões científicas divergentes.** `experiment/data/08_reporting/final_results_validation.md` descreve o portfólio antigo contendo PSO, DE, SARSA e GA-PPO e afirma EOQ global, SA perfil e ganho 6,2%. `kpis.parquet` contém EOQ, sS, Newsvendor, CappedBaseStock, BigDataNewsvendor, FixedInterval, VendorResponsive, GA, SA, DQN, PPO e GA-DQN. O SA atual custa 1.147,86, não 584 no perfil predominante. O arquivo de backup possui 18 políticas e não deve ser confundido com a matriz de junho. Não transportar 6,2%, 48% ou tabelas de qualificação para o novo artigo sem declarar versão e origem. A qualificação é base da hipótese e história do estudo, não garantia de identidade do experimento atual.

2. **Cinco repetições não são cinco treinamentos independentes.** Em `core/inventory_env.py::run_policy`, `base_seed` não é utilizado; o ambiente e a demanda são determinísticos, e `policy_fn` é reutilizada. Em `pipelines/inventory_simulation/nodes.py::_worker_generic`, a política é treinada/otimizada uma vez antes de repetir a avaliação. Todas as 1.740 linhas atuais têm `TIC_std` nulo (máximo abaixo do limiar numérico); `NS_std` é essencialmente ruído de ponto flutuante. Não afirmar 725 observações independentes por política nem 8.700 pontos independentes. A unidade é a série; para inferência sobre clientes/lojas, há agrupamento por loja.

3. **Separação temporal parcial, com informação futura residual.** `_split_demand` separa 17/21 ciclos no modo atual, mas `scale_parameters_per_store` usa `mu` e `sigma` de `scenarios_meta` calculados no horizonte completo para estoque inicial, limites de busca e máximos de ação. `InventoryEnv.__init__` calcula `_global_mu/_global_std` do bloco que está sendo avaliado; `_state` e a primeira previsão ingênua usam essas estatísticas futuras. O perfil é calculado sobre todos os 38 ciclos em `demand_profiling/nodes.py::compute_demand_features`. A exclusão de loja na seleção de política **não elimina** esses canais de informação. Não chamar o resultado de avaliação totalmente fora da amostra.

4. **Restrição é sobre serviço médio, não série a série.** No limiar 0,70 a melhor global tem 40 séries abaixo do alvo e o perfil cruzado tem 39 (26,9%). A média acima de 0,70 não demonstra factibilidade individual. As políticas por perfil podem ser mais caras do que a global apesar de seleção retrospectiva porque cada perfil precisa cumprir uma restrição distinta; os conjuntos factíveis são diferentes. Esse detalhe explica por que o ganho não é automaticamente garantido.

5. **Código atual não é manifesto do código que gerou os números.** As implementações de RL/GA e backend de previsão foram corrigidas; parâmetros documentam mudanças de operadores, objetivos, penalidades de risco e orçamento. Não atribuir causalmente custo alto de PPO ou SA ao algoritmo em geral nem comparar runtime/GPU sem novas execuções controladas. A matriz é evidência de implementações/configurações históricas específicas.

6. **Custos simulados não comprovam economia monetária realizada.** Custos configurados holding=1, stockout=5, ordem fixa=50, unitário=0,5 são parâmetros do simulador. A decomposição HoldingCost+StockoutCost+OrderCost confere numericamente com TIC (erro máximo 2,3e−12). Não converter o resultado para economia real em R$ sem justificativa empresarial/calibração. O pedido pago no fim do horizonte pode permanecer em trânsito; não há valor terminal ou warm-up explícito.

7. **Lead time exige convenção explícita antes de nova simulação.** `reset` cria `lead_time+1` posições, e `step` remove chegada antes de acrescentar novo pedido. Um pedido gerado no passo t entra fisicamente no estoque no passo t+L+1 sob essa indexação. Se o texto usa chegada em t+L, há divergência de um ciclo. O estado de decisão também precede a chegada desse passo. Não alterar código agora e reaproveitar os números antigos como se viessem da dinâmica alterada.

8. **Há métricas de seletor, mas não validação operacional convincente do PSE.** `policy_selector_metrics.parquet` contém acurácia de CV 0,44755±0,49724, enquanto a tabela por classe deriva de `predict(X)` após `fit(X,y)` em toda amostra. A acurácia aparente aproximada de 0,986 é de treino; não é desempenho em novas séries. `policy_selection/nodes.py` usa rótulo ótimo por série retrospectivo e CV por observação; é preciso validar agrupando loja e medir custo/serviço das recomendações, não apenas acurácia de rótulos.

9. **Tamanho e descrição da amostra devem ser auditados.** O texto antigo aproxima 40 lojas e 10 produtos, enquanto a chave atual tem 115 lojas; a tabela antiga diz 71% Lumpy, mas slides e `demand_profiles` registram 145/145 Lumpy. Use os artefatos computáveis atuais e declare que a seleção se restringe a esse regime. ADI/CV² é calculado no código com média e desvio sobre todos os valores, incluindo zeros; não apresentar como implementação idêntica à classificação original sem discutir essa definição.

10. **Nomes de perfis são descritores heurísticos.** Sparse High Impact é também a classe padrão residual, High Volatility Seasonal inclui burstiness sem sazonalidade e Unstable Trend inclui entropia alta sem tendência. Não afirmar que os nomes diagnosticam causalmente cada propriedade. As recomendações fixas em `dominant_policy` do perfil não são as políticas empiricamente vencedoras; a reanálise usa somente o nome do perfil, nunca essa coluna.

## Experimentos mínimos antes de uma contribuição preditiva submetível

1. Congelar versão de código, dados, ambientes e manifesto com hashes; definir horizonte, lead time, custo terminal e estoque inicial antes de olhar resultados.
2. Reconstruir metadados, normalização e perfis exclusivamente no treino; não estimar constantes da avaliação. Corrigir a convenção de chegada e validar trajetória com exemplo manual.
3. Fazer avaliação em origens temporais sucessivas, com validação interna para selecionar políticas/hiperparâmetros e teste intocado. Separar lojas para alegações de novas lojas. Manter global otimizada e EOQ, incluir políticas simples competitivas; avaliar todos sob o mesmo alvo de serviço.
4. Treinar métodos estocásticos com múltiplas sementes distintas, salvar resultados por semente e por série e reportar runtime/hardware/orçamento. Repetições determinísticas não precisam ser executadas cinco vezes.
5. Avaliar ganho de perfil contra melhor global por seleção aninhada, mais ablação sem perfil, permutação de perfis e escolha aleatória. Medir custo e NS nas recomendações, além de frequência de violações e custo de treinamento.
6. Usar bootstrap por loja e correção para múltiplas comparações quando comparar muitas políticas; reportar intervalos e todos os resultados, inclusive nulos. A matriz atual já permite sensibilidade e cross-fitting, mas não reconstrói incerteza de treinamento ou prevenção de informação futura.

## Matriz de alegações

| Alegação | Situação |
|---|---|
| Há heterogeneidade custo-serviço entre implementações preservadas | Sustentada descritivamente |
| Perfis atuais têm 116/18/11 séries | Sustentada pelos arquivos |
| Perfil supera a melhor política global em custo a 0,70 | Não sustentada; estimativa cruzada tem sinal contrário |
| Perfil economiza contra EOQ mantendo o mesmo serviço | Não sustentada; serviço cai 16,06 pontos percentuais |
| Média NS≥0,70 garante cada loja atendida | Falsa |
| PSE está validado nacionalmente ou em implantação | Não sustentada |
| Cinco replicações são independentes | Não sustentada pelo código/artefatos |
| Novo artigo apresenta reanálise condicional, com limitações explícitas | Sustentável |

## Revisão do experimento temporal complementar proposto

`experiment/scripts/analysis/sca_temporal.py` e `temporal_protocol.md` foram inspecionados sem alteração pelo revisor. A divisão 17/10/11 é cronológica, os perfis usam só calibração e as cinco heurísticas consultam inventário/pipeline/histórico atendido, não os normalizadores do estado do bloco de teste. A seleção para cada loja exclui todas as suas séries da validação das demais. Essa separação melhora substancialmente a integridade temporal da decisão em comparação com a matriz histórica. Não converte o estudo em avaliação nacional ou experimento de RL.

Pontos a resolver/explicitar antes da redação final:

- **Uma série tem demanda total zero nos 11 ciclos de teste.** `InventoryEnv.kpis` atribui serviço 0 por dividir 0 por epsilon. Fill rate é indefinido para essa série. Registrar a convenção e reportar sensibilidade; manter seu custo. Não contar silenciosamente essa ocorrência como falha em atender demanda existente.
- **EOQ histórico usa uma escala dimensional específica.** `EOQPolicy` calcula D=mu×len(calibração)=17mu, mas h é por unidade/ciclo. A fórmula EOQ habitual com essas unidades empregaria mu, não 17mu. A implementação histórica pede lotes sqrt(17) maiores. Identificar a variante exatamente e, se houver comparação de superioridade contra EOQ clássico, adicionar variante coerente de forma transparente em sensibilidade fixada antes dessa execução, sem ajustar pelo teste.
- **Newsvendor é uma heurística normal mu+z×sigma, com z=1,28.** Não é o quantil empírico ótimo obtido de uma razão crítica calibrada dos custos; a implementação se combina ainda com regra de ponto de pedido. O nome precisa de definição operacional.
- **VendorResponsive é uma formulação inspirada por uma descrição textual**, conforme o aviso em `core/policies_zabraoui.py`, não reprodução exata do artigo de Zabraoui. FixedInterval também representa uma interpretação (R,S) de uma descrição de intervalo fixo.
- O cálculo de ACF com lag 17 nos primeiros 17 ciclos retorna zero por construção; a classe High Volatility Seasonal ainda pode aparecer por burstiness, mas não existe evidência sazonal estimável nesse treinamento curto.
- As séries existentes constituem uma coorte selecionada retrospectivamente e compartilham nove produtos. Exclusão por loja e bootstrap por loja não removem dependência por produto nem viés de inclusão de séries futuras. Essas limitações já estão parcialmente no protocolo e devem aparecer no artigo.

Esses pontos foram comunicados ao agente principal antes de usar os resultados no manuscrito. Não solicitei escolher um protocolo que torne a hipótese positiva.
