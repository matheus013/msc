# Parecer editorial — concisão e legibilidade da dissertação

Revisão interna de 14/10/2026 sobre as fontes em edição. Não alterei capítulos, `tcc.tex`, fontes ou margens. O PDF-base informado pelo coordenador possui 118 páginas; reduzir aproximadamente 40% corresponde a cerca de 71 páginas. A contagem definitiva depende da compilação, não do tamanho dos arquivos. O plano antigo `PLANO_ENXUGAMENTO_TEXTO.md` parte de 124 páginas e decisões de portfólio já conflitantes: serve como histórico, não como meta ou protocolo vigente.

## Resultado da leitura

A introdução, os correlatos e as conclusões reescritos já usam frases mais claras. O maior potencial restante está na metodologia: na leitura inicial, `proposta.tex` tinha aproximadamente 77 KB e repetia arquitetura, ambiente, formulação de seleção e protocolo em seções separadas. As reduções devem remover explicações duplicadas, não parâmetros, resultados ou fontes. Base brasileira e M5 são eixos complementares centrais e precisam permanecer.

## Edições concretas prioritárias

| Local na fonte inicialmente lida | Edição proposta | Conteúdo a preservar |
|---|---|---|
| `proposta.tex`, abertura e Visão Geral, linhas 1–15 | Manter um parágrafo de propósito, um de execução e uma figura. Remover a descrição textual completa de cima para baixo que repete os nós do desenho. | Separação entre benchmark implementado e PSE ainda em validação. |
| `proposta.tex`, Instanciação do Ambiente, linhas 40–60, e Ambiente de Simulação, 345–371 | Fundir em uma única seção operacional. Remeter à equação conceitual do capítulo 2; informar no método apenas ordem de eventos, pipeline, demanda perdida e cobrança. | Estoque físico versus posição, lead time efetivo, estoque inicial/terminal, custos e diferenças entre conjuntos. |
| `proposta.tex`, Rede Varejista, linhas 61–71 | Trocar três escalões descritos em lista e parágrafo repetitivo por duas frases: a unidade é loja–produto; armazém é contexto de abastecimento. | Não insinuar otimização conjunta ou produto como escalão logístico independente. |
| `proposta.tex`, Métricas, linhas 73–95 | Uma tabela KPI/definição/papel, com custo e serviço formalizados uma vez. Remover repetição de que as métricas alimentam comparação, viabilidade e rótulos. | Distinção entre média simples e ponderada, viabilidade individual/grupal, tratamento de denominadores zero. |
| `proposta.tex`, definição AIPE, linhas 30–38, e PSE, 435–520 | Manter uma única formulação de menor custo viável e um quadro entrada/saída/estado da validação. Excluir três paráfrases da mesma equação e o segundo diagrama de seleção se reproduzir a arquitetura. | Condições de rotulagem, ausência de teste concluído do PSE, fallback, agrupamento e avaliação futura. |
| `proposta.tex`, Walk-forward, 372–421, e Execução Temporal, 551–fim | Um quadro REAL/M5/diagnóstico complementar com cortes, periodicidade, pré-processamento e estado de auditoria. Separar protocolo desejado de execução documentada. | Cortes temporais, vazamento possível, distinção entre previsão e controle. Não chamar ausência de vazamento de garantida sem confirmação. |
| `proposta.tex`, Artefatos e Protocolo, 528–fim | Um parágrafo e tabela de arquivos com função; retirar a terceira enumeração de dados→features→simulação→rótulo. | Hashes, parâmetros, versão, comandos e limitações de reprodução. |
| `conceitos.tex`, Métricas de Previsão, 173–200 | Preservar MASE/sMAPE/RMSSE e convenções. MBE/Theil cabem em uma frase caso não fundamentem tabela de resultados. | Fórmulas efetivamente usadas e escalas obtidas no treino. |
| `conceitos.tex`, famílias de políticas | Explicar genericamente cada família em poucas linhas; manter fórmulas fundamentais. Detalhes de camadas, passos, orçamento e hiperparâmetros pertencem ao método/apêndice. | Diferença entre regra e algoritmo que a calibra; EOQ com unidades coerentes; RL sem garantia automática de serviço. |
| `resultados.tex`, tabelas/figuras | Manter uma representação por achado: tabela completa para valores, figura para relação custo–serviço. Texto deve interpretar contrastes, não narrar cada célula. | PB, BA, M5, serviço, composição do portfólio e proveniência. |
| `conclusoes.tex`, síntese | Um parágrafo por resposta à pergunta de pesquisa; contribuições e limites em bloco curto. | Resultados negativos e status do PSE, sem reensinar as famílias de políticas. |
| `plano_continuidade.tex`, consolidação e cronograma | Cortar a narrativa que repete a tabela de etapas; tabela preserva critério de conclusão. Reservar texto a dependências técnicas relevantes. | M5 já existente, nova validação temporal e expansão brasileira distinta de evidência concluída. |
| Apêndices e glossário, sob responsabilidade do coordenador | Consolidar parâmetros comuns; retirar prosa de interpretação e infográficos que repetem corpo. Um glossário de siglas efetivamente usadas. | Nenhum PDF de referência deve ser apagado; valores de reprodução continuam rastreáveis. |

As linhas referem-se à versão lida antes das alterações dos demais revisores. Após a reescrita, localizar pelo título da seção.

## Repetições que ainda exigem decisão de edição

1. A advertência "PSE não validado" aparece na introdução, metodologia, resultados, conclusões e plano. Ela é necessária na introdução, no método e na conclusão; nos demais locais usar remissão curta e informar a limitação específica daquela análise. Não repetir todos os limites em cada capítulo.
2. Previsão é auxiliar: definir uma vez no referencial e lembrar na tabela de resultados de previsão. Retirar explicações completas repetidas após cada métrica, família ou estágio.
3. Descrição de pipeline: a arquitetura, a lista de estágios, a seção de artefatos e a execução temporal repetem quase a mesma sequência. Cada representação deve ter função distinta; a figura mostra fluxo, a tabela registra configurações e o texto justifica escolhas.
4. Custos, serviço e efeito chicote reaparecem nas listas de métricas e nas conclusões. Fórmulas no referencial; convenção implementada no método; valores e consequências nos resultados.
5. O contraste brasileiro/M5 deve ter uma síntese conjunta, não dois relatos completos com repetição do vocabulário. A diferença de serviço e de protocolo é o achado; não eliminar M5 para economizar páginas.

## Exemplos de linguagem mais leve

| Formulação pesada ou redundante | Alternativa |
|---|---|
| "Esta seção instancia, no ambiente experimental, a formulação conceitual..." | "O simulador aplica o modelo do Capítulo 2 a cada série loja–produto." |
| "A organização deste capítulo acompanha a arquitetura conceitual em camadas..." | "A Figura X mostra as etapas. A seguir descrevemos dados, políticas e avaliação." |
| "Essa estrutura influencia diretamente o AIPE..." com nova enumeração | "Cada série loja–produto é uma unidade de decisão; capacidade conjunta não é modelada." |
| "O terceiro grupo reúne três datasets analíticos... Em conjunto, esses artefatos..." | "A Tabela X identifica as entradas e saídas necessárias à reprodução." |
| "A validação produz estimativas de generalização temporal não viesadas" | "O protocolo pretende avaliar períodos futuros; a auditoria deve confirmar que todas as etapas usam apenas o passado." |
| "habilitando testes ... com poder estatístico adequado" | "Foram registradas cinco execuções; a independência e a variabilidade precisam ser verificadas." |
| "o framework aprende a recomendar" sem resultado do seletor | "O framework prevê um seletor supervisionado, ainda sem validação concluída." |

## Compatibilidade REAL e M5

O capítulo de resultados agora registra corretamente artefatos M5: 2.408 séries, dez lojas, 287 produtos, 93 ciclos e 28.896 pares série–política. Deve permanecer claro que o M5 já foi executado; a tarefa futura é auditoria/reprodução e validação adequada, não fazer o primeiro experimento público. A introdução e as conclusões podem dizer "os artefatos M5 ampliam a evidência disponível, com protocolo sob auditoria". Não afirmar que usar a mesma base reproduz automaticamente Zabraoui, nem comparar custos absolutos entre períodos, escalas e parâmetros diferentes.

O resumo e a conclusão devem citar ambos os conjuntos de forma breve. O esforço externo é central à dissertação. Uma tabela compacta de escopo REAL/M5 substitui várias páginas de justificativa sem perder a comparação.

## Critérios para aceitar o enxugamento

- Medir redução física no PDF em relação às 118 páginas, sem alterar fonte, margens ou reduzir texto a letras ilegíveis.
- Conferir que tabela de escopo, seleção, eventos do simulador, cortes e principais parâmetros continuam presentes.
- Verificar que PB/BA e M5 continuam no argumento e que número de séries não vira quantidade de replicações independentes.
- Preservar citações que sustentam definições ou comparações; não cortar fontes pelo tamanho da bibliografia.
- Conferir transições, referências cruzadas, tabelas e figuras após recompilar.
- Se a meta de aproximadamente 71 páginas não for atingida, informar a redução efetiva. Não perseguir o número removendo evidência necessária.

## Adendo após a reescrita metodológica

Reli a versão nova de `proposta.tex`, informada pelo revisor de método com aproximadamente 5.252 palavras frente a 10.411 anteriores. Ela concentra ambiente e decisão, apresenta uma única formulação de seleção e distingue validação proposta de execução auditada. As duplicações prioritárias da metodologia estão substancialmente resolvidas. O texto esclarece a fila $L+1$, a avaliação nominal repetida sem cinco treinamentos independentes, o PSE ainda proposto e os artefatos M5 existentes. Não há mais a garantia de generalização não viesada da versão anterior. As sugestões da tabela são histórico da intervenção e não exigência para repetir cortes já feitos.

O referencial também foi reduzido pelo revisor, de aproximadamente 5.103 para 2.921 palavras. A redução textual combinada tende a contribuir para a meta sem esvaziar a reprodução. Confirmar no PDF final se figuras, tabelas e quebras de capítulo ainda ocupam espaço desproporcional.

**Decisão editorial:** direção de reescrita adequada e consolidação metodológica realizada. Revisão final de legibilidade, comparação REAL/M5 e alcance depende do texto final e da contagem de páginas compilada pelo coordenador. Apêndices e glossário permanecem as alavancas sob responsabilidade do coordenador.

## Verificação da versão compilada — 66 páginas

O arquivo `build/tcc.pdf`, compilado em 14/10/2026 às 15:54, tem **66 páginas**. Em relação à base de 118 páginas informada, são 52 páginas a menos, redução de **44,1%**, superando a meta aproximada de 40%. Não há marcadores `??` nas páginas extraídas. O log registra um overfull de 7,73 pt no objetivo específico da introdução (linhas 59–60), um ajuste localizado de quebra de linha; não justifica mudar margens ou fonte.

Conferi visualmente a página física 48, com início da seção M5, e a última página de referências (66). Ambas estão legíveis, sem corte de conteúdo; M5 permanece como evidência existente complementar ao caso brasileiro. As referências finais são legíveis. Esta amostragem não equivale à inspeção visual de todas as 66 páginas, realizada à parte pelo coordenador.

A prosa ficou substancialmente mais leve. A oportunidade restante é agrupar advertências gerais sobre artefatos, cinco repetições e PSE em uma subseção de limitações, mantendo nas seções de resultados apenas a limitação que muda aquele contraste. Evitar novo enxugamento mecânico: o objetivo quantitativo já foi alcançado e cortar metodologia ou M5 agora prejudicaria o argumento.

Adendo do coordenador: a recompilação final continua com 66 páginas e não registra mais overfull. O ajuste localizado da introdução foi, portanto, resolvido. As observações do log anterior permanecem apenas como histórico.
