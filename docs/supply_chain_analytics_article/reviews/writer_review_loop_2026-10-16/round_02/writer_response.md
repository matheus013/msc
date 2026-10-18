# Resposta do escritor — rodada 2

Status: **pronto_para_revisao**. Sete reparos locais implementados. Não houve compilação, treinamento ou alteração de configuração experimental. A prontidão científica para submissão permanece **não**.

## Respostas item a item

| ID | Alteração efetiva e localização | Verificação |
|---|---|---|
| C-W01 | Último parágrafo de Trabalhos relacionados: define positivamente a contribuição como procedimento verificável com referência calibrada, autorização de intervenção e diagnóstico seletor/portfólio/viabilidade. Identifica os métodos constituintes e situa a aplicação com vendas perdidas sem restringir toda a proposta ao simulador local. | Texto conserva a ausência de confronto empírico com integrações próximas e não atribui aos antecedentes uma ausência não verificada. |
| MTH-01 | Simulador, portfólio e calibração: separados esforço/sementes, objetivos de treino e calibração da Capped. GA usa custo acrescido de déficit com peso 30 e escala max(C,1). DQN/PPO/GA–DQN usam custo negativo e penalidade terminal com peso 30 e escala max(p×demanda total do episódio,1). Capped usa essa escala fixa; riscos unitários Bahia e zero M5 são explicitados. | Conferência estática dos snapshots históricos descritos abaixo. Penalidade, custo reportado e regra de viabilidade de validação estão separados; nenhuma garantia de serviço é acrescentada. |
| MTH-02 | Substituição dos métodos e decisão de intervenção: retirado “ajustadas nas unidades de teste”. As candidatas são calibradas exclusivamente no prefixo de treino das séries avaliadas; trajetórias posteriores fornecem os diagnósticos de teste. | Cronologia explícita, mantendo distinção entre esforço experimental e aplicação pretendida. |
| MTH-03 | Custo, serviço e escolha: barras definidas como média aritmética por caso série–origem, após agregar sementes, excluindo serviço indefinido; meta média sem garantia por série. Avaliação e desenvolvimento do piloto: global definida por partição de loja reservada e origem. | Nenhuma contagem alterada. A família pode mudar entre lojas excluídas/origens e permanece comum às séries recomendadas na mesma partição/origem. |
| ED-01 | Custo, serviço e escolha: h, p, K e c definidos por unidade/ciclo/pedido; barras definidas. Simulador: mu e sigma definidos pelo prefixo de treino. Oráculos: C* identificado como custo do oráculo do portfólio ou do MILP, conforme comparação. | As duas equações destacadas continuam idênticas ao snapshot da rodada 1. Valores dos custos, fórmulas das políticas e expressão da eficiência foram preservados. |
| ED-02 | Main, imediatamente após os inputs Bahia e M5: referências explícitas à fronteira custo–serviço, à tabela por origem e à figura de distribuição/comparação M5. As chamadas interpretam o papel dos objetos sem repetir contagens. | Três rótulos têm agora chamadas no corpo. Os dois fragmentos compartilhados permanecem byte a byte intocados. |
| ED-03 | Exceção autorizada no fragmento aipe_consequencias_resultados.tex: cabeçalho Serviço → Atendimento e quatro células em fração com quatro casas. | 68,97%=0,6897; 69,37%=0,6937; 69,25%=0,6925; 71,72%=0,7172. Comparação automática confirmou exclusivamente essas cinco substituições no fragmento. Custos, contagens, legendas e demais trechos não mudaram. |

## Conferência de objetivos no código congelado

Bases dos snapshots lidos, sem importar/executar treinamento:

- `experiment/data/08_reporting/research_v2/bahia145_cpu16_2026-10-15/code_snapshot/experiment/src/simulation/`
- `experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15/code_snapshot/experiment/src/simulation/`

Funções e configurações conferidas em ambas:

- `research/protocol.py:calibration_config`: modo constrained e peso 30; alvo 0,70.
- `research/production.py:production_config/fit_production`: orçamento e encaminhamento à calibração; componente GA do híbrido usa GAPolicyOptimizer.
- `core/policies.py:_eval_static`: aptidão negativa do custo mais penalidade relativa a max(TIC,1).
- `core/rl_torch.py:_terminal_deficit_penalty`: déficit ×30×series_tic_ref; aplicação ao último passo em DQN e PPO.
- `research/hybrid.py:episode`: penalidade terminal na última transição.
- `core/inventory_env_torch.py:series_tic_ref`: max(p×soma da demanda,1); `_risk_terms`: escala fixa vezes frequência de ruptura×(1−serviço) e excesso positivo da amplificação acima de um.
- `core/policies_sota.py:_calibrate_batch`: Bahia chama constrained_cost com defaults unitários dos riscos; M5 piloto v2 encaminha pesos zero da configuração. Ambos usam a escala fixa da série.
- `core/inventory_env.py`: retorno de etapa corresponde ao negativo do custo operacional, sem incorporar essa penalidade ao custo reportado.

Os hashes dos arquivos lidos estão em `frozen_code_checked.json` para conferência pelo coordenador/revisor. Nenhum desses arquivos foi modificado.

## Pendências científicas mantidas

**C-E01:** confronto verificável com integrações próximas e confirmação independente continuam abertos. O reparo de posicionamento identifica a proposta, sem converter isso em demonstração de prioridade.

**MTH-04:** as extensões permanecem desenvolvimento exploratório do mesmo piloto; validação aninhada não foi apresentada como independência de desenvolvimento.

**MTH-05:** calibrações distintas, orçamento desigual e sensibilidades ausentes continuam limitando o alcance. Nenhum resultado futuro foi antecipado.

## Autocheck e integridade

- Equações destacadas e grupos de citações idênticos à rodada 1.
- Fragmentos Bahia e M5 por classe, bibliografia, figuras, highlights, dissertação e experimentos intocados.
- Terceiro fragmento contém somente a normalização autorizada da tabela.
- As novas expressões em linha tornam explícitas funções efetivas de calibração já implementadas; não introduzem métricas novas.
- Revisão focada: nenhuma reformulação ampla de resumo, discussão ou conclusão foi feita nesta rodada.
- Compilação, conferência visual e decisão de fechamento ficam com coordenador e revisores.
