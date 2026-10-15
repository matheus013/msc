# Artigo para Supply Chain Analytics

Manuscrito de trabalho em português, 15 páginas, sem apêndice. Nenhuma submissão foi realizada.

**Pergunta central:** selecionar a política por perfil melhora o compromisso entre custo e atendimento frente à seleção global, usando a mesma informação passada?

O artigo se concentra nessa pergunta. A dissertação contém a formulação ampla do AIPE e as questões complementares. O [plano de evidências](../PLANO_DE_EVIDENCIAS.md) registra a ligação entre perguntas, análises e lacunas.

- `manuscript_ptbr.tex` / `manuscript_ptbr.pdf`: fontes e PDF atual.
- `protocolo_bahia_20261015.tex`: desenho experimental, métricas e fluxo da informação.
- `bahia_20261015.tex`: resultados completos das 145 séries Bahia em duas origens.
- `figures/`: fluxo, custo × atendimento, distribuição de atendimento e escolhas por perfil.
- `article_sources_ptbr_draft.zip`: pacote das fontes utilizadas, incluindo figuras.

Os 4.350 ajustes da Bahia estão completos. A seleção por perfil aumenta o custo em 1,93% e o atendimento médio em 3,89 pontos percentuais; o efeito concentra-se na primeira origem. Há 283 casos com atendimento definido e sete casos sem demanda no teste. A coorte é retrospectiva e os testes compartilham um ciclo.

O M5 responde à mesma pergunta em dados públicos. Sua matriz está em execução; valores parciais ou resultados de outros protocolos não preenchem esse espaço. Versões anteriores, `historical_ptbr.tex` e todos os PDFs permanecem preservados para rastreabilidade, sem integrar o manuscrito atual.

Compilação a partir da raiz:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/literature/build_sca_article.py --ptbr
```

As figuras são reproduzidas por `experiment/scripts/documents/article_research_figures.py`, com a matriz completa e as escolhas auditadas na validação. A rotina não treina políticas nem altera a execução M5.

Antes da submissão, ainda são necessários consolidação M5, revisão humana, confirmação de autoria e declarações e adequação ao idioma editorial. Dados ficam fora do Git e seguem o [backup no Google Drive](../operations/BACKUP_GOOGLE_DRIVE.md).
