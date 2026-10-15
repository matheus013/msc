# Revisão editorial interna do manuscrito completo

Data: 14/10/2026. Fonte inicialmente avaliada: `manuscript.tex`, versão com duas figuras, quatro referências e sensibilidade EOQ em unidades de ciclo; scripts `sca_validation.py` e `sca_temporal.py` consultados para conferir a descrição do seletor. O adendo abaixo registra a versão revisada com seis referências e PDF compilado de 16 páginas. Este é um parecer interno dos autores, não parecer da revista.

**Decisão: rascunho científico honesto, com revisões necessárias antes de submissão.** O texto responde à hipótese com resultados negativos e distingue replay histórico, transferência entre lojas, teste temporal e sensibilidade acrescentada após auditoria. Não encontrei reivindicação de um PSE treinado, implantação real ou garantia de aceitação. Há pendências de reprodução e confirmações humanas que impedem chamar o pacote de pronto para submissão.

## Resolvido em relação à revisão inicial

- O resumo apresenta a pequena diferença histórica, a ausência de vantagem fora da loja e no teste temporal, a queda de serviço frente ao EOQ legado e o efeito da correção dimensional.
- Os 145 pares loja–produto, 115 lojas, três perfis históricos, cinco heurísticas temporais e doze rótulos históricos são distinguidos. Contagens de séries não são confundidas com repetições independentes.
- O artigo informa a convenção efetiva de lead time, estoques reiniciados, ausência de crédito terminal, zero demanda no teste e custos ilustrativos no diagnóstico.
- Os intervalos são explicitamente condicionais; não refazem seleção nem medem incerteza de treinamento. Intervalos abrangendo zero não são interpretados como equivalência.
- Há seis keywords. O resumo informado pelo coordenador tem 245 palavras, dentro do limite de 250; contar novamente após alterações.
- Declarações de autoria, financiamento, conflitos e dados permanecem explicitamente pendentes. A declaração de IA não finge verificação humana concluída.

## E-F1 — definição dos perfis — resolvido no apêndice A

**Gravidade: maior para a versão submetida.** Seção 4 apresenta nomes e contagens, mas não define as regras de $g(i)$. O leitor não consegue reconstruir a intervenção avaliada sem buscar o código. Inserir uma tabela ou apêndice com features, fórmulas, thresholds, prioridade das regras e default. Distinguir rótulos históricos preservados das regras efetivamente usadas no prefixo temporal. Explicar que a coluna heurística `dominant_policy` não foi usada como escolha de política no experimento.

Também corrigir a expansão "High Volume Seasonal": a regra do código usa autocorrelação ou burstiness, e seu comentário a chama "High Volatility Seasonal". Preferir rótulo literal `High_Vol_Seasonal` ou esclarecer a denominação legada; não atribuir alto volume como critério não implementado. As regras definem um perfil mesmo sem autocorrelação sazonal calculável, o que torna esta precisão relevante.

Critério de encerramento: definição suficiente para reimplementar $g(i)$ e uma linha explicando se a proveniência exata dos perfis históricos é recuperável.

## E-F2 — tie-break e fallback — resolvido no apêndice A

**Gravidade: menor, mas necessário à reprodução.** Seção 3 remete ao script para os desempates. A implementação escolhe menor custo e depois maior serviço entre candidatos viáveis; se nenhum é viável, maior serviço e depois menor custo. O fallback de perfil ausente usa o conjunto global. Explicitar essas regras no texto e distinguir fallback por ausência de perfil de fallback por inviabilidade; a ordenação final residual pode permanecer no código versionado.

Critério de encerramento: frase completa no método e indicação de quantos casos exigiram fallback nas análises principais, ou remissão a tabela suplementar claramente identificada.

## E-F3 — delimitar a reivindicação bibliográfica — parcialmente resolvido

**Gravidade: maior para uma submissão competitiva, não erro factual demonstrado.** Quatro artigos da própria revista oferecem contexto dirigido, mas não sustentam uma revisão abrangente de seleção de algoritmos, demanda intermitente, classificação e desenho temporal. O texto evita reivindicar prioridade, o que é correto. Antes de submissão, ampliar a busca primária e incorporar os trabalhos fundamentais e comparadores mais próximos realmente verificados. Não aumentar o número de citações apenas por estética e não tratar o acervo disponível como literatura exaustiva.

Critério de encerramento: posicionamento da pergunta de seleção por perfis frente a literatura externa à revista, além do benchmark próximo já discutido; referências completas com números de artigo.

## E-F4 — localizar o material que acompanha o texto

**Gravidade: maior para submissão, menor para rascunho local.** "The local analysis bundle" e "full sensitivity tables accompany the analysis artifacts" são compreensíveis para o projeto, mas não identificam suplementos para um revisor externo. O pacote final deve ter Supplementary S1/S2 ou equivalente, com configuração, tabelas de sensibilidade, manifesto, comando de reprodução e limites de acesso. Um DOI de depósito ainda não existe e não deve ser inventado.

Critério de encerramento: enumeração dos arquivos suplementares e identificação estável de versão; declaração de dados final aprovada pelo autor.

## E-F5 — confirmações humanas continuam obrigatórias

**Gravidade: bloqueio de submissão.** A autoria conhecida pode permanecer no rascunho, mas coautoria, afiliação final, correspondência, contribuições CRediT, financiamento, conflitos, permissão de uso/publicação e compartilhamento de dados precisam de decisão humana. A declaração de IA deve ser atualizada após a efetiva conferência dos autores. Não enviar os atuais textos "await confirmation" ao sistema editorial.

## Registro da verificação inicialmente pendente

O template numérico editável atende à forma esperada, mas é necessário confirmar referências resolvidas e ordenadas, integridade das duas tabelas, legibilidade das figuras e ausência de texto cortado no PDF. A revisão editorial não autentica independentemente todo o histórico de controladores nem substitui os pareceres de método e novidade. Um resultado negativo pode ser publicável, mas este parecer não prevê a decisão editorial.

## Adendo — fonte revisada e PDF final de 16 páginas

Versão inspecionada: PDF modificado em 14/10/2026 às 15:39:12, 626.053 bytes. Não foram encontrados marcadores `??` no texto extraído. Conferi visualmente as tabelas 1 e 2 (páginas 8 e 10), ambas as figuras (página 11) e declarações/referências da última página (16). Tabelas e referências estão dentro da página e legíveis. As seis referências aparecem numeradas; os números de artigo estão presentes.

E-F1 está **resolvido**: o apêndice define features, regras priorizadas, default, diferença de CV² em relação a Syntetos e a impossibilidade de inferir sazonalidade com o prefixo atual. Ele separa regras temporais de rótulos históricos preservados, sem fingir regeneração do passado. A denominação foi corrigida para High Volatility Seasonal, e o mapeamento heurístico prévio de políticas é explicitamente ignorado.

E-F2 está **resolvido**: os desempates e ambos os tipos de fallback são descritos, e o contraste temporal principal é identificado como 144 escolhas iguais e uma diferente. A sensibilidade com seis políticas repete efetivamente a seleção com EOQ corrigido, informa 144/1 e continua sem vantagem agregada de perfis. A extensão é apresentada como emenda de auditoria, sem alegar pré-especificação prospectiva.

E-F3 está **parcialmente resolvido**: Rice (seleção de algoritmos) e Syntetos (demanda) acrescentam fundamentos externos à revista; o texto não reivindica revisão sistemática ou prioridade. Recomenda-se ampliar a cobertura contemporânea dos comparadores para uma submissão competitiva, mas o número seis de referências não é, por si só, uma violação editorial.

**Ajuste visual detectado:** no painel (a) da Figura 1, rótulos do eixo logarítmico de custo se sobrepõem e alguns nomes de políticas ficam próximos. Remover rótulos de ticks menores e limitar ticks principais, além de afastar os rótulos EOQ/$(s,S)$. A Figura 2 está legível. Este ajuste é de apresentação, não altera os resultados.

E-F4 está **parcialmente resolvido**: o README agora identifica `evidence/`, manifesto e scripts, além do ZIP plano de fontes. Há material local para acompanhar a revisão. A forma definitiva dos suplementos e o depósito externo permitido dependem da confirmação dos direitos dos dados.

**Pendências vigentes:** corrigir o eixo da Figura 1; finalizar identificação de suplementos e repositório permitido (E-F4); finalizar declarações humanas e autorização de dados (E-F5). As questões resolvidas acima são mantidas como histórico de revisão, não como bloqueios ainda abertos. A redação atual é um rascunho íntegro e verificável localmente, com limitações científicas declaradas; não é uma submissão autorizada ou uma garantia de aceitação.


## Verificação final de apresentação pelo coordenador

Figura1 regenerada com ticks principais fixos (200,500,1000,3000,6000), rótulos menores desativados e posições separadas das políticas próximas. Inspeção visual final confirmou correção dos rótulos sobrepostos. PDF recompilado com bibliografia,16páginas, sem referências indefinidas ou caixas excedendo margem. Os oito testes de integração passaram. Os pontos científicos e humanos pendentes permanecem conforme parecer.
