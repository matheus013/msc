# Parecer editorial interno — artigo em português

Data: 14/10/2026. Fonte revisada: `manuscript_ptbr.tex` e estrutura bibliográfica associada, após incorporação das matrizes REAL e M5. Este parecer não é revisão da revista. Não alterei texto, fontes ou PDF.

**Decisão atual: rascunho em português claro e cientificamente cauteloso, com lacunas editoriais de definição dos perfis e desempates resolvidas.** O texto mantém a rede brasileira como caso central e trata M5 como contraste já existente. Não promete ganho de perfil, transferência temporal do PSE nem reprodução exata de Zabraoui. Resultados modestos e negativos estão no resumo e na conclusão, acompanhados de serviço. Permanecem confirmações humanas e fechamento dos materiais de submissão.

## Pontos aprovados

- O título descreve as duas bases sem prometer um seletor treinado. O idioma português é explicitamente apresentado como versão de trabalho, não como versão pronta para enviar à revista.
- O resumo informado pelo coordenador tem 236 palavras; há seis palavras-chave. Não há ressalvas extensas sem resultado principal.
- As coortes, prefixos, doze rótulos, custos internos de cada base e orçamentos distintos são descritos. O leitor não é incentivado a comparar custos absolutos entre Brasil e M5.
- O fato de 91,90% das séries M5 não terem venda no prefixo é salientado sem conclusão causal indevida. Dez lojas não são confundidas com 2.408 unidades independentes.
- Os intervalos são condicionais, com resampling de lojas sem refazer seleção; perfis de histórico completo e controles históricos permanecem limitações explícitas.
- Viabilidade média e atendimento individual são separados. Custos menores com menor serviço aparecem como trade-off. Fallback é distinguido de viabilidade efetivamente atendida.
- Declarações humanas são marcadas como provisórias, e a declaração de IA não finge revisão humana concluída.

## Histórico dos ajustes e estado atual

### P-E1 — definição reproduzível dos perfis — resolvido

Gravidade: maior para versão submetida. A seção de perfis informa prioridade e famílias de sinais, mas não os limiares, fórmulas, ordem completa e default. Esse é o objeto da hipótese do artigo. Incluir um apêndice compacto ou remissão explícita a um suplemento numerado que contenha o protocolo; a definição já existente no artigo inglês pode ser adaptada, distinguindo regras do código atual e rótulos históricos efetivamente armazenados. Explicar CV² com zeros versus tamanho de demanda positiva. Não tratar nomes interpretáveis como categorias validadas.

### P-E2 — desempates e ausência de perfil — resolvido

Gravidade: menor. Descrever escolha de menor custo/maior serviço entre viáveis, regra residual estável e fallback global quando o perfil não existe nos dados de escolha. Atualmente há fallback por inviabilidade, mas não toda a regra de reprodução. Identificar script/versão em material suplementar em vez de apenas "mantidos no projeto".

### P-E3 — consolidar limitações repetidas sem removê-las

Gravidade: ajuste de estilo. O prefixo M5 nulo aparece no resumo, dados, resultados e discussão; essas posições cumprem funções diferentes. Nas limitações, usar remissão curta ao diagnóstico de coorte, sem reproduzir números e explicação. Fazer o mesmo para PSE não validado, perfis de histórico completo e não reprodução de Zabraoui: definir no método, interpretar no resultado, sintetizar na conclusão. Evitar cortar o alerta específico dos dez agrupamentos, que altera diretamente a inferência.

### P-E4 — padronizar vocabulário

Gravidade: menor. Na primeira ocorrência, definir "nível de serviço por volume (fill rate)" e depois usar consistentemente um termo. "Atendimento médio" pode ser confundido com volume ponderado; lembrar que é média simples dos fill rates. "Transferência entre lojas" é correto no contexto condicional, mas não usar apenas "validação externa" sem o qualificador de matriz histórica. Manter nomes ingleses das políticas como identificadores, sem traduções alternadas que pareçam outros controladores.

### P-E5 — saída editorial e declarações

Gravidade: bloqueio humano de submissão. Autoria, correspondente, CRediT, financiamento, interesses e direitos dos dados ainda precisam de confirmação. A versão final deve individualizar as declarações exigidas e substituir marcadores provisórios. Acrescentar suplemento identificável, comandos e manifesto de versões; organizar depósito permitido. A redação em português atende ao pedido atual, mas não comprova cumprimento da exigência de idioma do veículo; verificar/adaptar no fechamento editorial.

## Verificação visual

O coordenador compila o PDF português e realiza QA. Antes da entrega, verificar referências numéricas resolvidas, acentos, largura da tabela de coortes, cabeçalhos e ticks das duas figuras. No artigo inglês anterior houve ticks logarítmicos densos; conferir que o problema não reaparece na Figura dual 1. Este parecer de fonte não certifica a versão visual ainda em compilação.

Adendo visual: conferi o PDF de 16 páginas, modificado em 14/10/2026 às 15:59:49. Não há `??` no texto extraído. Inspecionei visualmente as Figuras 1 e 2 (páginas 11 e 12) e a última página bibliográfica (16). Figuras legíveis, ticks sem sobreposição e referências numeradas presentes. Acentos do corpo renderizam corretamente; problemas na extração textual não representam defeito visual. Não há corte evidente nessas páginas. A Figura 1 distingue seletores e referência EOQ com escalas por base, sem sugerir custo absoluto comum.

## Resultado final

A nova versão responde ao pedido de integrar base brasileira e M5 de maneira compreensível. Consolidar prosa de limitações melhora a leitura, mas não deve esconder incerteza ou enfraquecer o caso externo. A conclusão permanece adequada aos resultados disponíveis; não há garantia de aceitação editorial.

## Adendo após as correções finais

Reli a fonte corrigida enviada pelo coordenador. **P-E1 está resolvido:** há tabela das cinco regras com prioridade, limiares e default, além de fórmulas e convenções de ADI, CV² com zeros, burstiness, tendência, entropia e autocorrelação. A característica é vinculada ao cenário completo, sem fingir disponibilidade exclusivamente passada. As categorias continuam declaradas locais e não validadas como impacto econômico ou sazonalidade real.

**P-E2 está resolvido:** o texto explicita custo/serviço entre viáveis, ordem de desempate alfabética residual e seleção global para perfil ausente. O fallback por inviabilidade permanece claramente separado de viabilidade satisfeita.

A tabela do portfólio completo agora apresenta as doze políticas nas duas bases com denominadores de custo próprios. O resultado brasileiro em 0,95 inclui redução de 17,246%, sua queda de atendimento e os casos de fallback; o ganho não foi omitido nem transformado em atendimento garantido. A incerteza sobre a configuração histórica e os limites de recuperação do software continuam declarados. Essas mudanças melhoram equilíbrio e rastreabilidade editorial sem alterar o alcance da análise retrospectiva.

P-E3/P-E4 permanecem sugestões opcionais de fluidez e padronização, não bloqueios científicos. **P-E5 permanece aberto** por exigir decisões humanas sobre autoria, direitos, financiamento, interesses, CRediT e verificação responsável, além do idioma e depósito final. A QA visual da nova versão ampliada será feita pelo coordenador; a inspeção de 16 páginas acima pertence à versão anterior e é mantida apenas como histórico.
