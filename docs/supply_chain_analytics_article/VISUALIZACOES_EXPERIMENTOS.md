# Visualizações integradas aos experimentos

As cinco análises são produzidas pelo nó `generate_experiment_diagnostics`, presente em `reporting`, `profile_analysis`, `final_report` e nos benchmarks Bahia, M5 e Brasil. O pipeline `experiment_diagnostics` permite executá-las separadamente, reutilizando os resultados existentes.

| Arquivo PNG/SVG | Análise |
|---|---|
| `01_fronteira_custo_servico` | Custo e fill rate médios por política, meta de 70% e fronteira de Pareto, com cada base em seu painel. |
| `02_politicas_por_perfil` | Participação das políticas selecionadas por perfil na meta principal. |
| `03_distribuicao_fill_rate` | Distribuição por série das estratégias global e por perfil, com meta, quantidade abaixo dela e casos indefinidos. |
| `04_heatmap_perfil_limiar` | Política modal por perfil e limiar, participação da vencedora e proporção de fallback. |
| `05_caracterizacao_demanda` | ADI, CV² positivo, proporção de zeros, extensão total do histórico e quantidade sem demanda no prefixo. |

Os limiares padrão são 70%, 80%, 85%, 90% e 95%. As políticas vêm dos artefatos da execução; o histórico tem 12, enquanto novas rodadas podem usar o portfólio reduzido. Cada gráfico acompanha tabelas ou atributos exportados e um manifesto de escopo.

## Execução

A partir de `experiment/`, sem treinar políticas:

```powershell
python run.py run --pipelines experiment_diagnostics
python run.py run --pipelines experiment_diagnostics --env m5
```

Saídas em `experiment/data/08_reporting/experiment_diagnostics/`, `experiment/data/08_reporting/m5/experiment_diagnostics/` e, no ambiente nacional, `experiment/data/08_reporting/bot/experiment_diagnostics/`. Configuração em `reporting.diagnostics`: ativação, nome da base, meta principal, limiares e diretório.

O executor `research-production` também gera as cinco figuras depois da etapa Bahia/M5. A fila censitária permanece interrompida; adicionar as análises não a reiniciou.

## Interpretação

Nos resultados planos do Kedro, a seleção exclui a loja avaliada, mas utiliza o mesmo bloco de avaliação disponível. As figuras são identificadas como análise retrospectiva; não representam seleção temporal independente.

Nas rodadas de pesquisa com blocos separados, a escolha usa a validação das outras lojas e a avaliação usa o teste da loja excluída. Somente casos com todas as políticas, sementes declaradas e ambos os blocos concluídos entram na comparação. As sementes são agregadas dentro da série–origem; origens repetidas não aumentam o número de séries independentes.

A fronteira representa dominância nas médias agregadas, não superioridade em todas as séries. A matriz perfil–limiar apresenta a escolha modal entre as dobras, pois diferentes lojas podem receber políticas diferentes. O fill rate indefinido em teste sem demanda continua ausente e é contado explicitamente.

CV² é calculado nas demandas positivas, com variância amostral; difere do CV² de todos os ciclos usado nos perfis legados. ADI e CV² ficam indefinidos no prefixo sem demanda. Cada histograma informa quantas observações têm o atributo definido e usa os mesmos intervalos entre bases. Nas rodadas de pesquisa, distribuições descrevem os painéis elegíveis e as barras de prefixos zerados usam todas as candidatas do manifesto. No Kedro e no histórico, a cobertura é a coorte fornecida ao relatório.

## Prévia com os resultados preservados

As cinco figuras históricas, com painéis Bahia/M5, estão em `experiment/data/08_reporting/sca_article/dual/visualizations/`. São uma reanálise exploratória dos resultados antigos, com as limitações registradas no manifesto, e podem ser regeneradas com:

```powershell
python run.py historical-visualizations
```

As figuras da execução parcial ficam em `experiment/data/08_reporting/research_v2/census_2026-10-14/visualizations/`. Há 13 casos completos da Bahia e 12 do M5 nessa execução; a cobertura limitada de lojas impede algumas dobras e os gráficos registram a ausência de seleção válida.

Foram exportados PNG e SVG. O backend PDF local depende de um binário fontTools bloqueado pelo controle de aplicativos do Windows; os formatos gerados permitem revisar as figuras e utilizar PNG no LaTeX sem alterar a política de segurança.
