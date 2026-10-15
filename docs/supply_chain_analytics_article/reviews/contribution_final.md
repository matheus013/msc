# Revisão final independente — contribuição e posicionamento

Data: 2026-10-14. Revisão original: `manuscript.tex`, versão com reanálise histórica, diagnóstico temporal 17/10/11 e sensibilidade EOQ por ciclo; bibliografia então com quatro entradas. **O adendo ao final atualiza os pareceres para a versão com seis referências, apêndice operacional e seletores com seis políticas. Nos pontos alterados, prevalece o adendo.** Esta revisão não edita o manuscrito e não certifica os cálculos ou a implementação dos controladores.

**Decisão sugerida: não submeter ainda como artigo completo.** O texto atual é um rascunho cientificamente mais defensável, com resultado negativo e limitações expostos corretamente. Sua contribuição atual é uma auditoria de caso e uma demonstração empírica de fragilidade de comparadores e validação. Falta estabelecer o que esse caso acrescenta à literatura de seleção de políticas e se a conclusão resiste a um benchmark corrigido. Um resultado negativo pode ser publicável; o problema não é o sinal do resultado, mas sua identificação e alcance.

## O que a revisão resolveu

- Retirou a reivindicação de PSE supervisionado validado e de inovação dos híbridos.
- Separou o ganho incremental contra seleção uniforme do ganho aparente contra EOQ legado.
- Tornou explícitos o compromisso custo–serviço, serviço médio versus individual e serviço ponderado por demanda.
- Não misturou o antigo ganho de 6,2% com o portfólio atualmente analisado.
- Expôs a ausência de variação entre repetições, dependência dos dados históricos, bootstrap condicional e limitações da separação por loja.
- Adicionou um diagnóstico temporal e relatou que a correção dimensional do EOQ reduz custo e aumenta serviço em relação aos seletores. Não tentou recuperar artificialmente uma hipótese positiva.
- Limitou a conclusão às regras e ao caso avaliados. Não declarou equivalência estatística quando o intervalo inclui zero.

## Questões maiores remanescentes

### C1 — a bibliografia não estabelece a lacuna específica

Localização: Introdução, Related work e as quatro entradas de `references.bib`.

As quatro fontes são artigos da revista-alvo. Elas justificam relevância temática e oferecem comparadores próximos, mas não situam o problema na literatura de seleção de algoritmos, seleção de políticas de inventário, classificação de demanda e avaliação orientada à decisão. Não existe uma exigência de número mínimo de citações nesta revisão: a lacuna é temática, não quantitativa. Acrescentar referências apenas para aumentar o total não resolveria o problema.

O texto deve explicar, com estudos primários pertinentes, quais seletores já foram avaliados, com que separações de treino/teste e quais efeitos do serviço e do benchmark já eram conhecidos. É preciso distinguir a evidência nova — estas regras de perfil falharam neste caso sob este protocolo — de princípios metodológicos conhecidos, como evitar vazamento e usar comparadores adequados. O próprio manuscrito não reivindica prioridade universal; isso é correto, mas não substitui o posicionamento positivo de sua contribuição.

**Critério de encerramento:** revisão dirigida de seleção de políticas/inventário e seleção de algoritmos, com comparação verificável de decisão, representação, baseline e validação; uma frase de contribuição específica que sobreviva a essa comparação.

### C2 — o diagnóstico com baseline corrigido ainda não avalia a seleção no portfólio corrigido

Localização: Complementary temporal diagnostic, Table 2 e Conclusion.

O EOQ por ciclo foi avaliado fora do portfólio original e domina os seletores em médias de custo e serviço (167,06 e 0,8956, contra 219,35 e 0,7409 para perfil). Isso é uma sensibilidade útil e uma crítica forte ao benchmark legado. Não determina, contudo, como os seletores global e por perfil se comportariam se a política corrigida pudesse ser escolhida na validação. Ela pode alterar a política uniforme, escolhas de grupos e fronteiras de viabilidade.

É correto manter o protocolo original e a emenda separados. Para uma afirmação sobre o valor da seleção depois da correção, deve existir uma segunda análise explicitamente exploratória, com portfólio corrigido e seleção repetida sem escolher políticas pelo teste. A interpretação do teste já consultado deve permanecer exploratória; uma nova janela intocada ou amostra externa seria evidência mais forte.

**Critério de encerramento:** incluir a correção em um portfólio de seleção coerente, preservar o histórico das emendas e mostrar se o efeito incremental persiste. Enquanto isso, nomear o achado como falha do seletor no portfólio legado e sensibilidade ao comparador, sem convertê-lo em rejeição geral da seleção contextual.

### C3 — falha da representação não está separada de falta de oportunidade para selecionar

Localização: RQs, Discussion e descrição de perfis.

O diagnóstico usa um prefixo de 17 ciclos que torna a característica sazonal de lag 17 zero por construção. Os perfis de teste apresentam grupos de tamanho um e dois, e a atribuição difere substancialmente da classificação histórica. Um resultado negativo nessas condições pode decorrer da representação, da escassez amostral, do portfólio, do ambiente de simulação ou da ausência real de vantagem de especialização. O texto reconhece limitações, mas ainda não discrimina essas explicações.

Uma comparação exploratória com classificação macro simples, regra simples por características e oportunidade retrospectiva de especialização ajudaria a localizar a falha. Um oráculo somente dentro do portfólio pode quantificar espaço de melhoria, desde que seja descrito como referência ex post, sem promessa de alcançabilidade e com serviço comparável. Ablação da sazonalidade/entropia e estabilidade dos perfis também seriam úteis. Não há obrigação de todos esses experimentos para qualquer formato editorial, mas o texto precisa justificar quais bastam ao escopo escolhido.

**Critério de encerramento:** demonstrar se a regra de perfis falhou apesar de existir oportunidade de especialização ou se o portfólio corrigido favorece quase sempre uma única política. Formular a conclusão conforme a resposta.

### C4 — uma janela curta e um cenário de custos limitam a força do ensinamento gerencial

Localização: Temporal diagnostic, Managerial interpretation e limitações.

Onze ciclos com estoque/pipeline reiniciados, ausência de crédito terminal, prazo implementado de três ciclos e coeficientes ilustrativos podem afetar ranking e serviço. A transparência atual é adequada. Todavia, uma auditoria demonstrativa desses efeitos em um cenário não sustenta recomendação ampla sobre adoção de perfis em varejo.

**Critério de encerramento:** conferir sensibilidade do sinal a condições de início/fim, prazo e custos; preferir múltiplas janelas futuras ou replicação pública. Se isso não for realizado, manter interpretação como evidência de caso e explicitar que o protocolo não isola o valor da segmentação de todas essas escolhas de modelagem.

## Suporte das quatro citações

| Fonte | Alegação usada no manuscrito | Julgamento |
|---|---|---|
| Tadayonrad e Ndiaye, DOI 10.1016/j.sca.2023.100026 | Relação entre avaliação de previsão, desempenho de inventário e segurança. | Sustentada pelo PDF publicado, pp. 1–2 e 12–13. Não usar como validação das equações ou do simulador deste projeto. |
| Pietukhov et al., DOI 10.1016/j.sca.2023.100041 | Analytics preditiva para KPIs de supply chain em caso empírico. | Sustentada: p. 1 descreve modelo logístico/neural e dados de maturidade Lean em 30 equipes. É contextual, distante de seleção de reposição e dispensável se não desenvolver essa distinção. |
| Zabraoui et al., DOI 10.1016/j.sca.2025.100154 | Benchmark multialgoritmo, GA–DQN e arquitetura modular já publicados. | Sustentada, pp. 1–2 e 16–19. O artigo também inclui GA–PPO; isso deveria ser mencionado se a discussão de novidade dos híbridos for ampliada. A referência não certifica a implementação local nem torna os experimentos diretamente comparáveis. |
| Theunissen et al., DOI 10.1016/j.sca.2026.100229 | Distinção entre retenção estatística e validade dos critérios decisórios. | Sustentada, pp. 1–2. O manuscrito evita afirmar que aplica T-SAV ou supera seus resultados; posicionamento apropriado. |

Metadados das quatro entradas correspondem aos PDFs e catálogo consultados. Não há afirmação falsa de resultado próprio sustentada por essas fontes. A insuficiência bibliográfica principal está nos estudos que faltam para a lacuna central, não no conteúdo dessas quatro citações.

## Ajustes menores que melhoram precisão

- No texto temporal, usar “best among the original five candidates” quando “best uniform” puder sugerir que inclui o EOQ corrigido. No histórico, “strong comparator” é uma qualificação relativa ao portfólio, não demonstração de otimalidade.
- Definir integralmente as regras e limiares dos perfis no artigo ou suplemento. Nomes interpretáveis não permitem reproduzir a classificação; “seasonal” pode ser acionado também por burstiness no código e não comprova sazonalidade.
- Reportar eventual dependência entre lojas que compartilham produtos como limitação adicional do bootstrap por loja. Isso não invalida automaticamente o contraste descritivo, mas restringe sua interpretação populacional.
- A falta de melhoria estatisticamente demonstrada não prova ausência de melhoria útil em toda a população. O texto atual geralmente preserva essa distinção; mantê-la no título, highlights e carta editorial.
- “Reporting standard” na discussão deve ser proposta prática motivada pelo caso, não padrão formal da revista ou resultado normativo validado.

## Encaminhamento recomendado

Manter este manuscrito como **rascunho de auditoria com resultados negativos**. Um próximo artigo completo pode se tornar defensável se: (1) resolver o posicionamento bibliográfico; (2) avaliar a seleção com candidatos corrigidos e oportunidade de especialização identificada; (3) testar estabilidade em protocolo mais amplo ou replicação externa; e (4) liberar material reproduzível suficiente, respeitando a confidencialidade. A evidência também pode recomendar abandonar a camada de perfis atual: esse é um desfecho científico legítimo.

Não é necessário transformar resultados negativos em positivos para publicar. É necessário mostrar que a pergunta foi estudada com um comparador adequado e que a conclusão acrescenta conhecimento identificável além da descoberta de falhas locais de implementação. No estado atual, esse limiar ainda não foi alcançado para recomendar submissão como full-length article.

## Adendo — revisão das correções implementadas

Versão novamente inspecionada em 2026-10-14. Este adendo substitui avaliações factuais superadas da revisão acima.

**C2 resolvido para o escopo de sensibilidade exploratória.** O texto agora relata seleção em um portfólio de seis políticas que inclui o EOQ com unidade corrigida. O seletor global escolhe esse candidato nas 145 séries (custo médio 167,06; fill médio 0,8956); o seletor por perfil o escolhe em 144 séries e Newsvendor em uma (167,59; 0,8929). Portanto, o perfil continua sem vantagem agregada de custo ou serviço após a correção. A emenda continua identificada como posterior à análise inicial. Não é necessário obter um efeito positivo nem repetir esse pedido de correção. Essa execução, feita após conhecer o teste, é uma sensibilidade exploratória e não uma nova validação prospectiva.

**C1 parcialmente resolvido.** A bibliografia passou de quatro para seis referências, incorporando Rice (1976), *The Algorithm Selection Problem*, e Syntetos, Boylan e Croston (2005), *On the categorization of demand patterns*. O problema geral de seleção e a origem da classificação de demanda agora têm referências próprias. Isso corrige a ausência completa dessas bases. Não estabelece ainda, sozinho, uma lacuna empírica atual frente a outros estudos de seleção de políticas de inventário. O trabalho precisa de posicionamento dirigido adicional se quiser sustentar uma contribuição ampla de artigo completo. Não existe cota numérica de referências nem exigência de citar mais trabalhos da própria revista.

**Reprodutibilidade dos perfis melhorada e ajuste menor encerrado.** O novo apêndice define regras, prioridades, limiares, variáveis e desempates. Explicita que o CV² operacional inclui ciclos com zero, enquanto a categorização original citada usa variabilidade dos tamanhos positivos de demanda. Também declara que a regra denominada sazonal pode ser acionada por burstiness, e que o lag sazonal não é identificável no prefixo de 17 ciclos. A diferença da taxonomia original deixou de ficar implícita. Isso permite interpretar e reproduzir o procedimento; não constitui validação independente de seus perfis.

**C3 permanece como limite de interpretação, não pedido de fabricar vantagem.** Com as seis políticas, o contraste perfil versus global muda a decisão de apenas uma série. O caso oferece pouca oportunidade observada para estimar um efeito incremental estável. A conclusão adequada é a que o rascunho adota: estas regras, neste portfólio e protocolo, não demonstraram utilidade adicional. Ainda não é possível distinguir amplamente falta de informação dos perfis de falta de oportunidade de especialização, nem extrapolar para toda seleção contextual. A observação sobre um contraste muito esparso e massa do bootstrap em zero ajuda a calibrar essa interpretação.

**C4 e a contribuição geral permanecem em aberto.** Continuam reais a janela temporal única, poucos ciclos, reinicialização do estoque/pipeline, custos ilustrativos, coorte regional retrospectiva e resultados históricos de controladores não regenerados independentemente. Nenhuma dessas restrições invalida automaticamente a auditoria de caso; em conjunto, restringem uma contribuição geral ou uma alegação operacional de implantação. Uma extensão pode produzir resultado negativo, positivo ou nulo: o critério científico é robustez do teste e avanço identificável, não sinal favorável.

**Decisão atualizada:** manter como **artigo em rascunho que exige revisão maior**, com auditoria e resultados negativos claramente documentados; ainda não recomendar submissão como artigo completo pronto. O bloqueio de comparação com portfólio corrigido foi atendido, as referências fundamentais e as regras operacionais foram acrescentadas. A próxima decisão é editorial/científica sobre a suficiência desse estudo de caso, informada por posicionamento bibliográfico mais completo e, se necessário, ampliação temporal ou replicação independente. Não atribuir ao parecer uma exigência de confirmar a hipótese original.
