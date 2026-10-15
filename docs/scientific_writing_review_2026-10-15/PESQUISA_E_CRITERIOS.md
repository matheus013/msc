# Pesquisa sobre escrita científica e critérios da revisão

Consulta: 15 de outubro de 2026. Escopo: manuscrito em português para Supply Chain Analytics e dissertação em desenvolvimento. Esta pesquisa é uma síntese editorial dirigida ao projeto, não uma revisão sistemática da literatura sobre escrita.

## Fontes primárias e aplicação

1. [Elsevier — 11 steps to structuring a science paper](https://www.elsevier.com/connect/11-steps-to-structuring-a-science-paper-editors-will-take-seriously). Texto de orientação de um editor, explicitamente distinto de política editorial. Aplicação: revisar métodos e resultados antes do resumo; reservar a discussão para interpretação; evitar novidade exagerada e enumeração de todos os números no texto. Quantidades sugeridas de páginas ou referências não foram adotadas como requisitos da revista.
2. [Elsevier Researcher Academy — Structuring your article correctly](https://researcheracademy.elsevier.com/writing-research/fundamentals-manuscript-preparation/structuring-article-correctly). A página pública apresenta a função da estrutura e dos elementos de indexação; o curso completo exige cadastro e não foi tratado como conteúdo integralmente consultado. Aplicação: alinhar título, objetivo, resumo e conclusão.
3. [EASE — Guidelines for Authors and Translators, 2018](https://www.ease.org.uk/wp-content/uploads/2018/11/doi.10.20316.ESE_.2018.44.e1.pdf). Diretrizes e apêndices sobre resumo, ambiguidade, coesão e simplicidade. Aplicação: frases com sujeito identificável, termos estáveis, parágrafos com uma ideia central, resultados quantitativos no resumo e legendas compreensíveis. O PDF foi preservado em `sources/`.
4. [EASE — histórico e atualização das diretrizes](https://ease.org.uk/publications/author-guidelines-authors-and-translators/). A associação informa que o formato antigo deixou de ser atualizado em 2020 e aponta uma publicação ampliada de 2024. A página da atualização foi consultada, mas não foi preenchido formulário para obter conteúdo restrito. Não se apresenta o PDF de 2018 como norma atual da revista.
5. [EASE — Quick-Check Table for Submissions](https://ease.org.uk/publications/ease-statements-resources/quick-check-table-for-submissions/). Aplicação: distinguir revisão de redação, conferência de conteúdo e preparação efetiva para submissão. Uma lista de checagem não comprova qualidade científica ou aceitação.
6. [MIT — Writing a Research Report](https://web.mit.edu/21w732/www/Handouts/scientific-paper.htm). Aplicação: revisar conteúdo, estrutura e estilo nessa ordem; descrever métodos como processo científico; apresentar resultados antes de interpretá-los. Detalhes sobre arquivos e histórico de desenvolvimento só permanecem no corpo quando afetam a interpretação.
7. [MIT Communication Lab — Creating an Outline for Your Master’s Thesis](https://mitcommlab.mit.edu/aeroastro/commkit/creating-an-outline-for-your-masters-thesis/). Aplicação: ligar pergunta, afirmação, evidência e raciocínio entre capítulos. A dissertação precisa explicar o percurso de pesquisa, enquanto o artigo concentra uma pergunta. Sugestões locais de quantidade de citações não foram transformadas em metas artificiais.
8. [UFAL/SiBi — Normalização de trabalhos acadêmicos](https://sibi.ufal.br/portal/?page_id=1770) e [registro do Manual da UFAL, segunda edição, 2025](https://www.repositorio.ufal.br/handle/123456789/10627). Aplicação: conferir elementos pré-textuais e nomenclatura institucional. O registro confirma a edição; a leitura integral do PDF de 35,75 MB não foi obtida pela ferramenta de navegação. O formato institucional existente foi preservado. Não se declara conformidade ABNT integral sem conferir classe, normas do programa e elementos de depósito.
9. [Nature — Reporting standards and availability of data, materials, code and protocols](https://www.nature.com/nature/editorial-policies/reporting-standards). Referência complementar de transparência computacional: descrever acesso efetivo e restrições de dados/código. Seus requisitos são da Nature; não foram atribuídos à Elsevier. Aplicação: diferenciar reanálise reproduzível de reprodução do treinamento histórico.
10. [Supply Chain Analytics — Guide for authors](https://www.sciencedirect.com/journal/supply-chain-analytics/publish/guide-for-authors). O endereço oficial retornou HTTP 403 nesta consulta. Consequência: não foi inventado limite específico de palavras, quantidade de figuras ou exigência atual de submissão. O template Elsevier já documentado no projeto foi mantido; a conferência editorial específica permanece pendente antes da submissão.

## Skills procuradas

A lista oficial curada de `openai/skills` foi consultada pelo instalador oficial: não contém uma skill específica de revisão científica ou dissertação. Foi localizado e lido o [SKILL.md de scientific-writing, K-Dense](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-writing/SKILL.md), licença MIT, versão declarada 2.3. Uma cópia de consulta está em `sources/scientific-writing_SKILL.md`.

O fluxo propõe registro de evidências, conferência numérica, distinção entre redação e aprovação e revisão das declarações. É compatível como referência de processo; suas rotas clínicas não se aplicam à simulação de estoques. Não foi instalado nem executado software desse repositório. Não se declara cumprimento de sua certificação humana de cada afirmação. A revisão usa as ferramentas locais existentes; conferências automáticas não substituem avaliação dos autores e orientadores.

## Critérios operacionais adotados

- Cada seção responde a uma pergunta identificável e contribui para o argumento central.
- O resumo informa problema, desenho, amostra, achados principais e limite da conclusão.
- Resultados históricos, reanálise e protocolo futuro têm identificação e tempos verbais próprios.
- Coortes BA e M5 permanecem separadas; resultados do piloto não são atribuídos à expansão.
- Quantidade de séries, unidades, denominadores, serviço e referência de custo acompanham os contrastes.
- Uma economia com queda de atendimento é descrita como compromisso, não como dominância.
- Incerteza condicional, falta de independência e uso de informação futura não desaparecem com revisão textual.
- Ressalvas repetidas são concentradas onde afetam o argumento, preservando leitura fluida.
- Tabelas mantêm seus valores, com notação decimal em português e unidades que representem o cenário.
- Citações e referências cruzadas são conferidas; fontes novas de escrita ficam neste relatório, sem inflar a bibliografia científica do estudo.
- Os dois PDFs são recompilados e inspecionados visualmente. Versões anteriores ficam em `before/`.

## Limites desta intervenção

A pesquisa de escrita não é nova evidência experimental. Durante a revisão, o autor autorizou integrar a execução da Bahia concluída em 15/10. Por isso, foram conferidos os 4.350 ajustes e realizadas novas agregações e reamostragens condicionais a partir das saídas existentes; nenhum controlador foi retreinado. O script e os derivados estão em `bahia_completed/`, com hashes das fontes. A revisão não resolve, por redação, lacunas de rastreabilidade das matrizes históricas, nem representa aprovação institucional, aceitação editorial ou versão pronta para depósito/submissão.

## Conferência bibliográfica adicional

Foram consultados 55 DOIs distintos no Crossref; 36 responderam, e as demais consultas tiveram limitação de taxa, identificador escapado ou ausência de registro. O arquivo `bibliography_crossref_audit.json` preserva as respostas anteriores às correções. Resolução de DOI confirma metadados, não o conteúdo de todas as afirmações citadas. As divergências relevantes foram verificadas em fontes primárias:

- [Syntetos, Boylan e Disney, registro institucional da Lancaster](https://research.lancaster-university.uk/en/publications/forecasting-for-inventory-planning-a-50-year-review/): a revisão é de 2009, volume 60, S149–S160, DOI `10.1057/jors.2008.173`. A entrada local atribuía 2012 e um DOI de outro trabalho. Ano, volume, páginas, DOI e chave de citação foram corrigidos.
- [Preil e Krapp, página do artigo na Elsevier](https://www.sciencedirect.com/science/article/pii/S0925527322001670): artigo 108578, DOI `10.1016/j.ijpe.2022.108578`. A entrada local apontava 108595, correspondente a outro artigo.
- [Batsis e Samothrakis, versão publicada no repositório da Essex](https://repository.essex.ac.uk/38214/1/1-s2.0-S0957417424004068-main.pdf): autores, volume 249, artigo 123541 e DOI `10.1016/j.eswa.2024.123541` corrigidos. A entrada local tinha autores e identificador incompatíveis com o título.
- [Boylan e Syntetos, página do capítulo na Springer](https://link.springer.com/chapter/10.1007/978-1-84800-011-7_20): mantidas as páginas 479–506 da editora, apesar de registro institucional divergente; corrigido apenas o escape do sublinhado no campo DOI. Sublinhados de dois DOIs da Evolutionary Computation também foram normalizados para identificadores resolvíveis.

Não se afirma auditoria integral do conteúdo de todas as 61 referências citadas na dissertação. Referências não resolvidas automaticamente não foram excluídas nem consideradas falsas por esse motivo.
