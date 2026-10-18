# Parecer científico e alterações

Revisão de desenvolvimento em 16/10/2026. Objeto: artigo em português para Supply Chain Analytics e dissertação AIPE. Não é parecer humano nem previsão de aceite.

## Avaliação

O trabalho tem uma proposta metodológica identificável: organizar seleção, intervenção e diagnóstico de políticas de reposição sob contratos explícitos. A base empresarial brasileira sustenta a relevância aplicada, e o M5 fornece um referencial público. A contribuição precisa ser defendida pela articulação das decisões e pelos resultados que ela permite examinar. Reunir algoritmos conhecidos ou nomear uma arquitetura não estabelece originalidade por si.

A evidência quantitativa do manuscrito continua sendo condicional. A seleção por perfil melhora atendimento com custo maior na Bahia; o preditivo por classe falha no piloto M5; a extensão por consequências com recuo produz ganho pontual concentrado. Esse conjunto permite discutir quando a contextualização ajuda e onde falha. Ainda não demonstra melhoria generalizável do AIPE, eficácia de toda substituição modular ou ganho econômico de implantação.

## Alterações executadas

1. Reescritos resumo, introdução, discussão e conclusão do artigo, com problema, dificuldade decisória, mecanismo e alcance em sequência explícita.
2. Adicionada uma rota de leitura para a avaliação: perfil, estimação por série e causas das perdas. A extensão por consequências é apresentada como outra implementação do PSE na mesma matriz.
3. Reforçada a diferença entre a execução das interfaces e a utilidade operacional. Retiradas repetições de defesa da arquitetura e de detalhes da fila computacional.
4. Corrigida a inconsistência de quatro versus cinco etapas na introdução da dissertação. Também foi corrigida a remissão do referencial ao capítulo experimental.
5. Reescritos os resumos português/inglês da dissertação. Introdução, contribuição, motivação dos módulos e síntese das perguntas foram refinadas.
6. Mantidos equações, chaves bibliográficas, matrizes e fragmentos quantitativos. Não foram incorporados resultados parciais do M5 nem substituídos os resultados Bahia sem reconciliação global da versão harmonizada.
7. Atualizados os PDFs e o pacote portátil do artigo; fontes e PDFs anteriores foram preservados.

## Mapa entre afirmações e evidências

| Afirmação | Evidência identificada | Situação |
|---|---|---|
| AIPE define cinco etapas, interfaces e regra de intervenção | Método do artigo; capítulos 4 e 5 da dissertação | Sustentada como formulação e implementação |
| Controladores e seletores podem ser substituídos nas interfaces exercitadas | Sete candidatas; variantes por perfil, classe e consequências; matriz compartilhada | Sustentada no escopo testado; portabilidade integral não demonstrada |
| Perfil acrescenta 3,89 p.p. de atendimento com 1,93% de custo na Bahia | Matriz de 4.350 ajustes; 580 recomendações reconciliadas; fragmentos Bahia | Sustentada na matriz reportada |
| Preditivo por classe perde 13,9% em custo e 4,11 p.p. em atendimento no piloto | 1.350 ajustes, 90 casos série–origem e 86 casos com serviço definido; auditoria numérica | Sustentada no piloto |
| Consequências com recuo reduz custo em 1,56% e acrescenta 0,27 p.p. | Tabela das variantes: 1.070,03 e 71,72%; três intervenções; cerca de 92% da economia em uma série | Sustentada como resultado exploratório |
| Oráculos distinguem seleção, portfólio e inviabilidade física | Partição M5: 39 atendimentos, 21 erros de seleção, 15 lacunas do portfólio, 11 limites físicos e quatro casos sem demanda | Sustentada como diagnóstico retrospectivo |
| A base real demonstra economia empresarial realizável | Simulação com vendas possivelmente censuradas e custos de cenário | Não sustentada; excluída da redação |
| A metodologia é superior em geral ou inédita em toda articulação | Faltam confirmação independente e comparações diretas com antecedentes próximos | Não sustentada; excluída da redação |

## Autorrevisão em cinco dimensões

**Contribuição — o que é próprio do AIPE?** A formulação conecta evidência comparável, duas abordagens do PSE, intervenção e diagnóstico no problema de reposição. O texto situa os antecedentes de classificação e seleção. A força da distinção frente às integrações mais próximas ainda requer comparação mais aprofundada; a revisão editorial não a supre.

**Clareza — o leitor entende o objeto antes dos algoritmos?** A introdução define a escolha da família e a referência global calibrada por série. O método apresenta responsabilidades e interfaces antes das implementações. Cinco etapas agora aparecem de modo consistente na dissertação e no artigo. Cada parágrafo revisto foi conferido por função e relação com os vizinhos; o esboço reverso está registrado separadamente.

**Força experimental — os ganhos são consistentes?** Não. Há troca de custo por serviço na Bahia, deterioração preditiva no M5 e melhoria concentrada com recuo. O texto expõe essas diferenças e não transforma pequenas médias favoráveis em superioridade estabelecida. Os intervalos da extensão alcançam zero.

**Completude — o que falta para a submissão?** Consolidar o protocolo ampliado, distinguir desenvolvimento e confirmação, verificar calibrações, aprofundar comparadores próximos e sensibilidades relevantes. Uma coorte maior não constitui automaticamente confirmação independente. Autoria, declarações e revisão humana também permanecem abertas.

**Solidez — quais hipóteses limitam a conclusão?** Vendas podem ser censuradas; custos não são monetariamente calibrados; horizontes são curtos; sementes/origens repetem unidades; estado comum e oráculos são condicionais; os grupos impõem restrições de serviço diferentes das individuais. Esses limites foram conservados nas seções pertinentes.

## Verificações

- Artigo: 22 páginas, resumo de 199 palavras pelo verificador, seis figuras, 17 referências citadas e pacote portátil validado.
- Dissertação: 62 páginas, redução de 47,5% frente às 118 da versão original; compilação sem diagnósticos pendentes.
- Auditoria das matrizes passou, sem novos treinamentos ou alterações nos experimentos.
- Páginas renderizadas e folhas de contato de ambos os PDFs inspecionadas. Não se observaram recortes, sobreposições ou referências indefinidas nas verificações realizadas.
- As skills usadas orientam processo e critérios de revisão; sua instalação não demonstra qualidade científica nem garante aceite.

Skills aplicadas: research-paper-writing, academic-codex-bridge, academic-paper-reviewer e scholarly-writing-refiner. As três últimas são adaptações locais identificadas; as skills de outros domínios foram instaladas, mas não impostas a este trabalho.
