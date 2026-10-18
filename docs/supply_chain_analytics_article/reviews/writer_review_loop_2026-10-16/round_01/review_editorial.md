# Revisão independente de clareza e apresentação — rodada 1

Manuscrito congelado: `round_01/source/manuscript_ptbr.tex`.
SHA-256: `250a815d12b444dbc36ebcff15a2e8def56e40f8fb8d9989b42d6c231eefd6e3`.
Parecer baseado no texto completo, nos três arquivos de resultados e no PDF compilado de 22 páginas. Não consultei os outros pareceres ou a resposta do escritor. Não editei o manuscrito, código ou execução de experimentos.

**Decisão editorial: minor_revision.** O texto já é suficientemente claro para circular como manuscrito de trabalho. Não encontrei problema crítico ou maior de redação/apresentação. Há três correções locais verificáveis antes do fechamento do ciclo.

**Prontidão científica para submissão: não.** A extensão por consequências permanece exploratória, o próprio texto reconhece confirmação independente pendente, o posicionamento frente às integrações mais próximas requer demonstração adicional e faltam tradução/declarações autorais finais. Esta conclusão não transforma lacunas experimentais em falhas de estilo.

## Forças e leitura como pesquisador externo

A pergunta é legível: a heterogeneidade cria uma oportunidade de seleção, cujo valor precisa ser medido frente a uma família calibrada por série. O AIPE é apresentado como metodologia, com cinco etapas e contratos; os controladores e o PSE são implementações substituíveis. O leitor distingue referência global, perfil e estimação por série antes dos detalhes experimentais.

A progressão introdução → antecedentes → procedimento → avaliação → Bahia → M5 → discussão é consistente. O M5 abriga os diferentes seletores na mesma seção 6; a subseção 6.4 é identificada como substituição de implementação e conserva o estatuto exploratório. O texto explica os denominadores e não transforma sementes/origens em observações independentes. Os resultados negativos são visíveis e a concentração da economia permanece no resumo, resultado e discussão.

ADI, CV, GA, DQN, PPO, RL, EOQ e MILP são definidos; AIPE/PSE são apresentados por extenso no resumo e no corpo. Os identificadores dos controladores são apresentados junto a seus comportamentos. Não encontrei sigla central cujo entendimento dependa do histórico do chat. Os problemas de primeira definição remanescentes são símbolos matemáticos, descritos em ED-01.

## Achados

### ED-01 — minor — writing

Local: manuscript_ptbr.tex:87–95,133,155 (seções 3.3,4.2,4.4).

Evidência: A equação apresenta h,p,K,c e só informa seus valores. Mu e sigma aparecem nas fórmulas EOQCycle sem definição textual. C* na eficiência é implícito e pode indicar o oráculo do portfólio ou o ótimo MILP.

Reparo: Definir os quatro componentes de custo por unidade/pedido/ciclo; definir mu e sigma como estatísticas do treino; explicitar que C* é o custo da referência retrospectiva considerada, identificando cada oráculo. Acrescentar que barras representam médias entre casos definidos. Preservar valores e fórmulas.

Verificação: conferir a definição/chamada/escala no próximo snapshot e no PDF compilado, sem modificar números auditados. Estado: open.

### ED-02 — minor — writing

Local: aipe_bahia_resultados.tex:3–29; m5_pse_20261015.tex:19–26.

Evidência: Os rótulos fig:cost_service, tab:ba2026_selection e fig:m5_comparison aparecem somente em suas definições: Figuras 2 e 4 e Tabela 3 não possuem chamadas explícitas no corpo. Os parágrafos interpretam resultados, mas a ligação precisa ser inferida pelo leitor.

Reparo: Acrescentar referências explícitas aos três objetos nos parágrafos que já interpretam seus resultados, indicando a comparação relevante. Não repetir os dados apenas para introduzir a chamada.

Verificação: conferir a definição/chamada/escala no próximo snapshot e no PDF compilado, sem modificar números auditados. Estado: open.

### ED-03 — minor — writing

Local: aipe_consequencias_resultados.tex:8–12 e tabelas 3–5 do PDF.

Evidência: Tabelas 3 e 4 apresentam atendimento como fração (0,7146); Tabela 5 muda para percentagem (71,72%) e também troca o título para Serviço. São valores equivalentes, mas a comparação imediata entre variantes exige conversão mental.

Reparo: Uniformizar a escala e o nome da coluna de atendimento nas três tabelas, preferencialmente fração com quatro casas para preservar o padrão existente; alternativamente converter todas para percentagem identificada no cabeçalho. Não alterar valores substantivos nem confundir diferença percentual com pontos percentuais.

Verificação: conferir a definição/chamada/escala no próximo snapshot e no PDF compilado, sem modificar números auditados. Estado: open.

## Inspeção visual

Renderizei e examinei as páginas 5, 12, 16 e 17 a 100 dpi, cobrindo figura conceitual, fronteira custo–serviço, decomposição de falhas e comparação por consequências/Tabela 5. Os arquivos estão em `editorial_visual/`. As figuras têm texto legível em visualização ampliada, eixos identificados e nenhuma sobreposição/corte relevante. A tabela de variantes cabe na página; nomes longos quebram ordenadamente. O gráfico de falhas distingue categorias e preserva contagens. Não há motivo para refazer figuras apenas por preferência visual. A figura conceitual possui três painéis de motivação e uso, enquanto o corpo descreve cinco etapas metodológicas; sua apresentação como síntese da motivação evita confundir esses níveis.

As legendas informam denominadores e restrições interpretativas. O problema remanescente de conexão figura/texto é especificamente a falta de chamadas explícitas descrita em ED-02, não ausência de análise dos resultados.

## Pendências que não se resolvem por redação

Confirmação independente das extensões do piloto; demonstração da vantagem incremental e do posicionamento frente às integrações próximas; calibração harmonizada se houver comparação ampliada; confirmação humana de autoria, financiamento, interesses e acesso a artefatos. Não é necessário alongar o artigo para repetir essas pendências: elas já estão reconhecidas, e os reparos editoriais não devem removê-las.

O fechamento editorial pode ocorrer após as três correções locais e sua verificação. Preferências opcionais de estilo, como encurtar ainda mais o resumo ou substituir todo termo técnico estrangeiro, não são bloqueios neste parecer.
