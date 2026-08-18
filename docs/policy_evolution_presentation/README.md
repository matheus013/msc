# Evolução do Portfólio de Políticas (Julho -> Agosto/2026)

Apresentação Beamer (LaTeX), mesmo tema e paleta da apresentação de
qualificação (`docs/qualification_presentation/`), documentando o antes e
o depois de cada uma das políticas de reposição do AIPE entre a versão
usada no Experimento 1 (julho/2026) e a reimplementação concluída em
agosto/2026 (commit `66d5ad8`, "reimplementa politicas em PyTorch e amplia
portfolio para 18").

## Estrutura

```
main.tex          # documento principal (tema, cores, \section{} por bloco, \input das seções)
sections/
  00_titulo/                # capa
  01_visao_geral/            # por que reimplementar, os dois defeitos, panorama geral
  02_simulador/               # InventoryEnv -> BatchInventoryEnv, equivalência numérica
  03_classicas/                # clássicas mantidas + SOTA clássicas novas + Zabraoui novas
  04_metaheuristicas/         # GA (antes/depois detalhado), SA, PSO, DE
  05_fitness/                  # evolução da função de aptidão em 3 estágios (+TR/BE)
  06_rl/                        # DQN, PPO (defeito do gradiente), SARSA
  07_hibridas/                  # GA-DQN / GA-PPO
  08_infra/                     # roteamento de GPU, hiperparâmetros alinhados ao Zabraoui
  09_fechamento/                # portfólio final 12->18, próximos passos
Makefile / .latexmkrc      # compilação via latexmk + xelatex (idêntico ao outro deck)
```

## Origem do conteúdo

Todo o antes/depois vem diretamente do código-fonte e do histórico git do
projeto -- nenhum número ou comportamento foi inventado:

- `git show 66d5ad8^:simulation/src/simulation/core/policies.py` -- estado
  de julho/2026 (GA/SA/PSO/DE via DEAP e scipy; DQN/PPO/SARSA via rede
  NumPy manual).
- `git show 66d5ad8` -- mensagem de commit da reimplementação (motivação,
  os dois defeitos corrigidos, validação de equivalência numérica).
- `simulation/src/simulation/core/{inventory_env_torch,
  metaheuristics_torch,rl_torch,policies_sota,policies_zabraoui}.py` --
  estado atual, incluindo os comentários e docstrings que já documentam
  cada decisão de fidelidade/ressalva metodológica.
- `simulation/conf/base/parameters/simulation.yml` -- diff de
  hiperparâmetros entre as duas versões.
- `simulation/AJUSTES_INFRA_2026-08-18.md`, item 21 -- extensão da
  aptidão com termos de risco TR/BE (mudança mais recente, feita na
  sessão de acompanhamento de infraestrutura).

## Regra de estilo deste deck

Cada slide de comparação usa os rótulos `\antes` (vermelho) e `\depois`
(verde) definidos em `main.tex`, e `\novo` para políticas que não
existiam em julho. A cor indica a natureza da mudança:
vermelho = corrige um defeito; verde = política nova; azul-petróleo =
reescrita/vetorização sem mudança de defeito conhecido.

## Compilação

Requer XeLaTeX (tema `metropolis` via `fontspec`) e `latexmk`.

```bash
make        # compila main.pdf
make clean  # remove artefatos de compilação
```

## Escopo

Este deck é sobre **implementação**, não sobre resultados numéricos novos.
Nenhum KPI comparando a versão de julho com a atual é reportado aqui --
essa comparação depende das reexecuções em andamento (Bahia v3, "bot",
M5), tratadas apenas como contexto no slide final.
