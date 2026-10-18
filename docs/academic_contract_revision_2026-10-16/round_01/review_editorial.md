# Revisão editorial independente — artigo e dissertação

**Decisão: artigo approve; dissertação minor_revision.** Nenhum achado crítico/maior de redação identificado. Um reparo localizado de siglas permanece na dissertação. Esta é avaliação nova dos documentos atuais, não extensão da aprovação do ciclo anterior.

**Prontidão científica para submissão/depósito: false.** Requer confirmação independente, consolidação experimental, confrontação da contribuição e validação humana/autoral conforme o escopo. Aprovação editorial não resolve essas pendências.

## Versão efetivamente lida

Conferi todos os arquivos de `manifest.json` contra SHA-256 real: **nenhuma divergência**. Artigo: `e918dff752219856d8ce62754403e4e10da681a4fa81b3b8ee6b1ad607dc7d06`; raiz da dissertação: `7912edd25149d40b3c8ebbcfe089a7453531b594454015cb2b4073d066185768`; manifest agregado: `4c414eaa3eace3739b6e8e4fcb5a7b8d531ac263e85c639547644e25dcc0e097`. Os hashes das entradas da dissertação constam desse manifest, necessário porque a raiz tcc.tex isolada não caracteriza os capítulos.

Li CONTRATO.md, writer_response.md, novas fontes de resumo/abstract, introdução, proposta, instanciação, conclusão e continuidade, artigo completo e os capítulos de fundamentos/relacionados para verificar definições. Não consultei pareceres dos colegas desta rodada, não editei fontes e não interagi com experimentos.

## Contribuição, harmonia e leitura

O novo destinatário é identificável: apoio ao gerente para avaliar custos evitáveis e atendimento, com decisão final humana. O texto diferencia intenção de uso, implementação parcial e benefício efetivamente medido. As lojas identificam contextos e dependência, sem ranking de desempenho dos gerentes. A referência global é metodológica e calibrada por série, sem ser tratada como operação real da empresa. O contrato de dados distingue observações, informação, estimação e cenário e impede transformar preço/receita em custos logísticos ou fotografia posterior de estoque em estado histórico.

O objetivo de reduzir custos preservando serviço aparece como finalidade, enquanto Bahia com custo maior/serviço maior e M5 exploratório delimitam o que foi demonstrado. Não encontrei reinterpretação dos resultados para alegar sucesso geral da finalidade. A recomendação pretendida inclui parâmetros e alternativas; a entrega atual permanece declaradamente parcial, sem ordens de compra executáveis ou validação com usuários.

As cinco etapas são estáveis entre introdução, método e conclusão. As seis dimensões temporais são distintas e coerentes: período básico, agregação, revisão PSE, horizonte, janela histórica e atualização dos modelos. A revisão por ciclo é contexto informado da empresa; a família fixa por sete ciclos é a configuração efetivamente avaliada. Revisar família, emitir pedidos e retreinar são explicitamente separados. Estado e pedidos em trânsito sequenciais permanecem requisito de avaliação futura.

Resumos PT/EN da dissertação preservam o mesmo destinatário, cinco etapas, resultados e valores, três intervenções, concentração de economia, custos de cenário, família fixa por sete ciclos e limites. Não há promessa extra na tradução. O resumo do artigo permanece seletivo e contextualiza corretamente Bahia/piloto M5. O leitor não precisa conhecer a conversa para identificar sua proposta.

## Achado EDC-01 — minor — escrita — open

**Local:** capitulos/introducao.tex:82; instanciacao_aipe.tex:27,47,148; capitulos/conceitos.tex:75,139,164,169; capitulos/plano_continuidade.tex:25.

**Evidência:** ERP, IQR, OLS e MILP aparecem sem expansão no corpo e sem entrada na lista; CPU/GPU também. No referencial, SARSA, MAPE, RMSE e KPIs são usados sem glosa suficiente da sigla. MBE é explicado como viés médio, mas a expansão inglesa permanece implícita. GA/DQN/PPO constam da lista anterior ao corpo e não constituem bloqueio equivalente.

**Reparo verificável:** Preferir termos por extenso para siglas de uso único: sistemas de gestão empresarial, intervalo interquartil, mínimos quadrados ordinários, indicadores. Expandir MILP e CPU/GPU na primeira ocorrência; glosar SARSA/MAPE/RMSE onde introduzidos, sem ampliar discussão tutorial ou modificar fórmulas. MBE pode receber expansão inglesa local ou manter a descrição de viés médio. Verificar compilações e resumos sem alterar quantitativos.

Texto mínimo sugerido:

- ERP → “sistemas de gestão empresarial”.
- IQR → “intervalo interquartil”.
- OLS → “mínimos quadrados ordinários”.
- “Seu MILP mantém” → “Um programa linear inteiro misto (MILP) mantém”.
- CPU/GPU → definir unidades centrais de processamento e unidades de processamento gráfico no primeiro uso.
- SARSA → indicar estado, ação, recompensa, próximo estado e próxima ação, preservando a atualização já apresentada.
- MAPE → “erro percentual absoluto médio (MAPE)”. RMSE → “raiz do erro quadrático médio (RMSE)”. KPIs → “indicadores”.

As siglas comuns e a lista existente reduzem a gravidade; por isso este achado é menor, não bloqueio metodológico. Pode ser resolvido com substituições locais e recompilação, sem nova revisão ampla. Expansão adicional de GA/DQN/PPO na introdução é opcional porque já constam da lista anterior ao corpo.

## Gates mecânico e semântico adaptados ao PTBR

Executei varredura reprodutível dos oito arquivos alterados (incluindo abstract inglês), registrada em `editorial_mechanical_scan.json`. Contagens: M1 travessão/separador 1; M2 contraste 1; M3/M4/M8/M9 preenchimento 0; M6 aberturas 0; M5/M16 promoção 0; M10 promessa vaga 0; M11 passiva inglesa 0; M12 excesso de palavras padronizado 0; M13 qualificadores 0; M14 ordinais artificiais 0; M15 exclamações 5; M17 verbos figurativos 0; M18 abertura vazia 0; pares de precisão 0.

Hits justificados: M1 é separador de metadado da revista (`\journal`), não travessão retórico; M2 é a instrução factual de preservar a matriz “em vez de guardar somente a vencedora”; cinco “!” são especificações TikZ de cores, não exclamações no texto. Essas ocorrências não justificam edição. A ausência de padrões ingleses não é prova automática da qualidade de toda prosa portuguesa; apliquei leitura semântica adicional.

S1/S3 apontam EDC-01. S4/S5/S7/S9: argumento legível, decomposição estável, sem promessa diferente entre método e avaliação. S10 foi adaptado: números centrais no resumo/conclusão são síntese convencional, não duplicação a eliminar automaticamente. S18/S21: as restrições factuais permanecem necessárias, sem exclusão de limitações para tornar o texto mais persuasivo. Não transformei regras estilísticas absolutas da skill em bloqueios artificiais.

## PDF e apresentação

Examinei os dois contatos completos do artigo (24 páginas), contatos 1,3,5 da dissertação (66 páginas) e os detalhes já disponíveis de método e resultados. Sem clipping/sobreposição visível. Figuras quantitativas mantêm mensagens legíveis, tabelas cabem nas margens e não há novo apêndice no artigo. O artigo conserva seis figuras. Não exijo novo gráfico por preferência decorativa.

Conferi `pdffonts` dos dois PDFs: todas as fontes reportadas estão incorporadas. Algumas do artigo são Type 3, característica herdada da compilação, sem ilegibilidade visual observada; não torna necessário refazer figuras ou alterar o contrato nesta rodada. O coordenador informa compilação sem referências indefinidas/overfull; não recompilei em paralelo.

## Repetição e continuidade

Os novos limites aparecem em suas funções adequadas: contrato no método, alcance na avaliação e síntese nas conclusões. Não encontrei repetição burocrática que exija corte para o fechamento. A dissertação é mais ampla que o artigo sem obrigar este a conter todo o trabalho. As pendências continuam experimentais ou de uso gerencial e não podem ser declaradas resolvidas por estas mudanças textuais.
