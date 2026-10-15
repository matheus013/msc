# Organização — 14/10/2026

A raiz contém três pastas de trabalho: `data/`, `experiment/` e `docs/`. `.git/` e `.gitignore` são os metadados necessários ao versionamento.

## Dados

`data/data/` foi desfeito: as fontes originais estão em `data/source/`; cenários JSON e CSVs preparados foram movidos para `experiment/data/legacy/`. A fonte M5 ficou em `data/m5_walmart/`.

A segunda cópia de vendas, antes em `simulation/data/01_raw/vendas/`, tinha 27 arquivos com hashes idênticos aos originais. Foi retirada do projeto, mantendo a fonte única intacta. O inventário SHA256 e a cópia recuperável estão no backup externo.

## Projeto Python

O projeto Kedro passou de `simulation/` para `experiment/`. Os scripts e o experimento PB foram integrados pela entrada única `experiment/run.py`. PB e Kedro usam o mesmo núcleo de ambientes, políticas e previsores. A seleção de dispositivo é compartilhada e prioriza GPU.

Foram retirados o segundo conjunto de módulos de políticas e os conversores de vendas redundantes. O conversor comum DuckDB preserva UF, agrega duplicatas entre arquivos e trata o ciclo comercial sem confundi-lo com mês. O conversor JSON sem dependências externas foi mantido.

A LSTM PyTorch é o backend atual; o NumPy histórico permanece selecionável. Novas execuções podem produzir resultados diferentes. Não houve reexecução dos benchmarks completos ou alteração dos resultados apresentados na qualificação.

## Produção acadêmica

`docs/qualification_presentation/` é a referência atual dos resultados. `docs/master_proposal/` contém as fontes da proposta/dissertação. `docs/archive/sbpo/` e `docs/archive/sbpo_versions/` guardam o artigo descontinuado e versões anteriores. `docs/references/` preserva os artigos e o acervo anterior de revisão.

Todos os PDFs foram preservados, inclusive duplicatas de artigos. Caminhos ativos das figuras e tabelas LaTeX foram ajustados para `experiment/data/`.

## Recuperação e Git

Arquivos retirados, código anterior e inventários estão em `C:\Users\mathe\mestrado-cleanup-backup-2026-10-14/`, fora do projeto. Os dados continuam locais e ignorados. Retirar dados do índice não os remove dos commits anteriores. As figuras e textos já versionados continuam preservados, mesmo quando o diretório de resultados é ignorado para novos arquivos.
