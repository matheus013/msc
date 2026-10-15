# Reescrita do artigo e refinamento da dissertação

Revisão de 15/10/2026. Artigo: 17 páginas, antes 19. Dissertação: 52 páginas, antes 62; a referência inicial tinha 118. PDFs e fontes anteriores foram preservados em docs/archive/aipe_revision_2026-10-15/.

## Artigo

O AIPE passa a ser a contribuição central, com PSE por perfil e preditivo. Introdução, posicionamento, métodos, discussão, conclusão, resumo e destaques foram reescritos. O texto articula três perguntas: valor da escolha por perfil, utilidade do preditivo/perfil explícito e distância às melhores referências.

São cinco figuras vetoriais: conceito AIPE, portfólio Bahia, escolhas por perfil Bahia, comparação M5 e decomposição de falhas. O manuscrito não usa apêndice. A comparação da Bahia permanece separada do piloto M5; não se apresentam rankings de custos entre bases como efeito da demanda.

## Dissertação

A metodologia foi reorganizada e condensada, distinguindo controlador e seletor. PSE deixa de significar somente modelo supervisionado. Foram incorporados piloto M5, diagnóstico, recuo exploratório, oráculos e limites físicos. A rastreabilidade está integrada à metodologia; fontes de apêndices permanecem preservadas, sem inclusão no PDF atual.

Resumo português e Abstract inglês, introdução, conclusões e continuidade foram alinhados. A capa e os elementos institucionais identificam dissertação em desenvolvimento, sem declarar qualificação/defesa realizada. A data corresponde à versão de trabalho.

O texto de resultados M5 é idêntico nos dois documentos para impedir divergência numérica. A dissertação mantém fundamentos, literatura, formulação e esforço computacional, além do recorte do artigo.

## Conferências

- Artigo: resumo de 218 palavras, 13 referências citadas, cinco figuras vetoriais com fontes Type 1 incorporadas; destaques dentro dos limites verificados.
- Compilação dos dois PDFs sem referências indefinidas, rótulos duplicados ou caixas excedentes.
- Conferência numérica de 580 escolhas principais Bahia contra a matriz completa.
- Métricas das cinco estratégias M5 reproduzidas a partir das recomendações congeladas; extensão com recuo conferida separadamente.
- 4.350 ajustes Bahia, 1.350 ajustes piloto M5, 90 ótimos de planejamento e partição das falhas confirmados.
- Inspeção visual das páginas, com conferência detalhada de capa, conceito, tabelas e gráficos.
- Pacote LaTeX do artigo atualizado para incluir apenas fontes e figuras utilizadas.

## Limites que a revisão não oculta

O PSE preditivo não foi avaliado na Bahia. O piloto M5 é desenvolvimento e o recuo foi motivado após observar seus resultados. A calibração da CappedBaseStock difere entre bases; há diferenças entre restrições individuais e agregadas. Nada disso foi transformado em confirmação independente ou superioridade geral.

Não foram treinados controladores, retomados M5 ampliado/Brasil, submetidos documentos ou alterados compartilhamentos. Nenhum PDF de conhecimento foi excluído. A versão inglesa anterior do artigo está preservada; a tradução da versão atual ainda não foi realizada.

Antes da submissão: protocolo final independente, harmonização de calibrações, revisão dos autores, tradução e fechamento de autoria/declarações. Antes do depósito: confirmação dos elementos institucionais e revisão dos orientadores.

## Auditorias

Os arquivos numerical_audit.json, package_checks.json e document_qa.json documentam as verificações. Os scripts audit_aipe_documents.py e verify_article_package.py reproduzem as conferências sem treinamento.

O guia oficial da revista retornou HTTP 403 na consulta desta revisão. Foi mantido o template Elsevier existente; não se declara reconferência integral das exigências atuais. A referência Cawley e Talbot (2010) foi verificada na página primária do JMLR: https://www.jmlr.org/papers/v11/cawley10a.html.
