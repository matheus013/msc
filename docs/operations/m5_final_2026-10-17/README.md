# Consolidação final Bahia e M5 — 17/10/2026

A auditoria `audit_completed_matrices.py` é somente leitura: verifica cobertura integral de política/semente/bloco, duplicidades, custos, limites do atendimento e denominadores. Não treina controladores nem seletores. Seus resultados estão em `matrix_audit.json`.

O M5 completou 79.890 ajustes, 159.780 linhas de métricas e 5.326 casos série–origem. As 2.408 séries pertencem a uma coorte histórica fixa dentro de 14.370 identificadores FOODS candidatos da fonte; não representam todas as FOODS. São 885, 2.033 e 2.408 elegíveis nas origens 50, 64 e 78. O atendimento é indefinido em 188 casos de teste (69, 106 e 13 por origem); custo usa 5.326 casos e atendimento usa 5.138. Há dez lojas, não 5.326 observações independentes. As origens repetem séries e as sementes são agregadas dentro de cada caso/política.

A Bahia tem 4.350 ajustes, 290 casos, 145 séries e 115 lojas. Atendimento usa 283 casos, com sete indefinidos. As duas origens de teste compartilham um ciclo. As aplicações devem permanecer separadas.

## Entradas e geração

Somente depois de concluída a avaliação canônica, execute na raiz:

```powershell
experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_unified_final_results.py
```

O gerador exige estado `completed` nas duas execuções e reconcilia os resumos com as métricas por caso. Não reexecuta controladores, não ajusta seletores e não altera arquivos das matrizes canônicas. Os certificados do planejamento perfeito são contados; eficiências só usam os casos/realizações comparáveis e certificados. Um plano não certificado não é convertido em ótimo presumido.

Saídas:

- `evidence.json`: todas as nove variantes, contrastes, intervalos, danos, recuperações, concentração, oráculos, perfis por origem, classes dos rótulos de validação, uso do piloto, margens e tempos de ajustes efetivamente novos.
- `resultados_unificados_20261017.tex`, idêntico no artigo e na dissertação: duas tabelas principais e quatro figuras.
- `diagnosticos_unificados_20261017.tex`, na dissertação: tabelas de diferenças, oráculo do portfólio e planejamento com informação perfeita.
- Figuras novas em PDF vetorial via PGF e PNG, copiadas para os dois documentos: custo/atendimento; diferenças pareadas; decomposição de falhas; distribuição do atendimento.
- CSVs por base: políticas, resultados por origem e tempos sem reaproveitamento.

O gerador não altera `manuscript_ptbr.tex`, capítulos ou prosa: a integração deve substituir os inputs antigos, mantendo uma única versão dos resultados. Não reexecute os geradores históricos `audit_aipe_documents.py`, `build_aipe_results_figures.py`, `article_research_figures.py` ou `plot_dissertation_consequences.py` para atualizar essa evidência: estão vinculados ao piloto/à matriz anterior.

## Interpretação

Custos são unidades de cenário, não reais. Intervalos usam 5.000 reamostragens por loja (semente 20261017), preservando séries e origens; são condicionais aos modelos/controladores ajustados, sem nova estimação. A comparação com o oráculo do portfólio usa médias por semente. O planejamento perfeito é confrontado com as realizações das políticas escolhidas por semente: seus denominadores não são intercambiáveis. Eficiências condicionais de variantes com coberturas diferentes não constituem ranking direto.

A separação das 30 séries usadas no desenvolvimento do piloto é reportada para rastreabilidade, sem afirmar confirmação independente das demais. Estabilidade de importância de atributos não foi persistida pelos seletores canônicos; o relatório não inventa esse resultado.
