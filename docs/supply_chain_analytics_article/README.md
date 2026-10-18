# Artigo AIPE para Supply Chain Analytics

Manuscrito de trabalho em português. O AIPE é uma metodologia modular de apoio ao gerente de logística e estoques; o PSE realiza associação por perfil ou estimação por série. Não houve submissão.

## Evidência consolidada

As duas aplicações usam matrizes separadas: Bahia, com 145 séries, 115 revendedores, 290 casos série–origem e 4.350 ajustes; M5, com uma coorte histórica de 2.408 identificadores de alimentos nas dez lojas, 5.326 casos e 79.890 ajustes. A coorte M5 não corresponde a todas as séries FOODS. Atendimento médio exclui sete testes sem demanda na Bahia e 188 no M5; custo inclui todos os casos. Sementes, séries e origens não são observações independentes.

O piloto anterior orientou o desenvolvimento. A ampliação permite avaliar as implementações em mais casos, sem constituir confirmação independente automaticamente. Custos são de cenário, não economia financeira realizada. O módulo opcional de atributos adicionais tem protocolo e estudos próprios, separados da matriz de controladores.

## Arquivos e reprodução

- `manuscript_ptbr.tex` / `manuscript_ptbr.pdf`: texto e PDF canônicos.
- `resultados_unificados_20261017.tex`: tabelas e figuras quantitativas compartilhadas com a dissertação.
- `figures/aipe_conceito_v3_contexto_ptbr.pdf`: arquitetura com módulo opcional; versões anteriores preservadas.
- `article_sources_ptbr_draft.zip`: fontes e figuras efetivamente incluídas, com bibliografia e classe.
- `build_manifest_ptbr.json`: hashes do pacote corrente.
- `../operations/m5_final_2026-10-17/`: auditoria, evidência agregada e registros da atualização.

Na raiz, após conclusão da avaliação canônica:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_unified_final_results.py
& experiment/.venv/Scripts/python.exe docs/operations/figure_context_2026-10-17/build_article.py
& experiment/.venv/Scripts/python.exe experiment/scripts/documents/verify_article_package.py
```

Geradores antigos baseados no piloto ficam preservados para reproduzir aquela versão; não devem substituir os resultados consolidados. O texto em inglês `manuscript.tex` é histórico e não está sincronizado com esta revisão em português. Antes de submeter: avaliação independente, revisão humana, declarações e autoria, tradução e conferência editorial. Dados e modelos permanecem fora do Git e seguem o backup autorizado.
