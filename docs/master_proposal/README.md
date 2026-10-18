# Dissertação AIPE

Versão de trabalho em português. A contribuição central é uma metodologia modular e rastreável de apoio ao gerente de logística e estoques. As implementações do PSE e as aplicações Bahia/M5 avaliam condições de uso e limites dessa proposta; não representam avaliação de eficiência das lojas ou dos gerentes.

## Organização e evidência

- `tcc.tex` / `tcc.pdf`: documento raiz e PDF canônico, com formato institucional `ic.cls`.
- `capitulos/`: introdução, fundamentos, literatura, metodologia, resultados, conclusões e continuidade.
- `instanciacao_aipe.tex`: desenho experimental separado da metodologia geral.
- `resultados_unificados_20261017.tex`: tabelas e figuras de resultados compartilhadas com o artigo.
- `diagnosticos_unificados_20261017.tex`: resultados detalhados adicionais da dissertação.
- `figures/`: figura conceitual com contexto opcional e gráficos quantitativos; versões anteriores preservadas.
- `referencias.bib`: bibliografia.

Bahia: 145 séries, 115 revendedores, duas origens, 290 casos e 4.350 ajustes. M5: coorte histórica de 2.408 identificadores, dez lojas, três origens, 5.326 casos e 79.890 ajustes. A expansão nacional permanece cancelada. A ampliação do M5 não transforma o piloto de desenvolvimento em confirmação independente. Custo inclui todos os casos; atendimento médio considera 283 casos definidos na Bahia e 5.138 no M5.

Os fragmentos de resultados anteriores ficam preservados para reprodução histórica, sem serem somados à evidência corrente. O módulo opcional de dados adicionais possui estudos próprios em `experiment/data/context_extension/`, documentação em `docs/operations/context_rounds_2026-10-17/` e não exige novo treinamento de controladores.

## Compilação

Na raiz:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/documents/build_dissertation.py --sync-pdf
```

O comando usa pdflatex e bibtex e só atualiza o PDF canônico sem referências indefinidas ou caixas excedentes. `revisao_concisao/comparison.json` distingue contagem aproximada de palavras de páginas efetivas. A auditoria da atualização está em `docs/operations/m5_final_2026-10-17/`.

Custos não estão calibrados financeiramente e a avaliação é de desenvolvimento. Parâmetros e trajetórias ainda têm limitações de persistência; uso por gerentes e confirmação independente continuam necessários. A capa informa a data da versão, não uma data de defesa. Elementos institucionais, banca e aprovação permanecem sob responsabilidade dos autores e orientadores. Não se declara versão pronta para depósito. Nenhum artigo em PDF foi removido.
