# Revisão final de métodos — 14/10/2026

**Decisão: rascunho de pesquisa tecnicamente honesto e adequado para revisão humana; revisão maior antes de submissão.** A conclusão negativa sobre o seletor avaliado é compatível com as análises. Não há base para afirmar que o agente adaptativo/PSE foi validado nem para recuperar os ganhos da qualificação como resultados do experimento atual. Esta decisão não prevê a decisão editorial da revista.

Revisados: `manuscript.tex`, `temporal_protocol.md`, `experiment/scripts/analysis/sca_temporal.py`, `experiment/data/08_reporting/sca_article/temporal/results.json` e tabelas CSV de seleção. O SHA256 do script confere com o manifesto da execução temporal. As análises históricas foram revisadas em `methods_initial.md`. Não alterei o manuscrito.

## Pontos resolvidos desde a primeira revisão

- O manuscrito abandona a alegação de ganho de 6,2% no portfólio atual e informa a divergência entre versões.
- As cinco repetições determinísticas não são tratadas como observações independentes.
- A seleção cruzada de políticas é distinguida de validação temporal de controladores.
- O novo diagnóstico usa 17 ciclos de calibração, dez de validação e onze de teste, com exclusão de todas as séries da loja-alvo na seleção. As cinco heurísticas ignoram a normalização do estado calculada com o bloco futuro; suas decisões dependem de parâmetros da calibração, estoque, pipeline e histórico atendido.
- Uma série sem demanda no teste mantém seu custo e recebe serviço convencional igual a um. A média excluindo-a é reportada; nenhum caso sem demanda existe na validação, portanto a convenção não altera a seleção.
- A sensibilidade EOQPerCycle corrige unidades da quantidade pedida e fica explicitamente fora do portfólio original. Sua adição posterior à auditoria é declarada.
- Reset por episódio, ausência de crédito terminal, coorte retrospectiva e dependência por produto são reconhecidos. O texto não inventa calibração empresarial dos custos.

## Verificação numérica do diagnóstico temporal

No limiar principal 0,70:

| Estratégia | Custo médio | Serviço médio | Serviço ponderado | Séries abaixo do limiar |
|---|---:|---:|---:|---:|
| Uniforme | 218,9774 | 0,742229 | 0,662533 | 59 |
| Perfil | 219,3527 | 0,740863 | 0,660947 | 60 |
| EOQ histórico | 386,7636 | 0,943841 | 0,919938 | 4 |
| EOQPerCycle | 167,0587 | 0,895606 | 0,828307 | 19 |

- Perfil versus uniforme: custo aumenta **0,1714%**; intervalo da redução [-0,5662%; 0,0000%]. Serviço muda −0,1365 ponto percentual; intervalo [-0,4242; 0,0000] ponto percentual. Não interpretar o limite zero como evidência de equivalência ou superioridade estatística do uniforme.
- Perfil versus EOQ histórico: redução de custo 43,2851%, serviço −20,2977 pontos percentuais.
- Perfil versus EOQPerCycle: custo aumenta **31,3028%**, intervalo [19,1824%; 42,4434%]; serviço cai **15,4743 pontos percentuais**, intervalo [13,3022; 17,5549] pontos percentuais em magnitude. EOQPerCycle domina os seletores nos dois objetivos agregados; não necessariamente em cada série.
- Macro serviço sem o caso de demanda zero: uniforme 0,7404385; perfil 0,7390639. A sensibilidade de convenção não muda a interpretação.
- Perfil temporal: 90 Sparse High Impact, 52 Unstable Trend, dois Fast Moving e um Low Volume Stable. Não há High Volatility Seasonal porque a ACF de lag 17 não é estimável em uma calibração de 17 ciclos e o critério alternativo não se ativa nessa coorte.

Os números do manuscrito conferem com esses artefatos, respeitado o arredondamento.

## Correções textuais solicitadas ao redator

1. **Nome de perfil:** o nome da classe de 11 séries na matriz histórica é High Volatility Seasonal, não High Volume Seasonal. Padronizar sem sugerir que o perfil é definido por alto volume.
2. **Uma única escolha diferente no diagnóstico principal:** a seleção temporal por perfil usa VendorResponsive em 144 séries e Newsvendor em uma série Fast Moving; o uniforme usa VendorResponsive nas 145. A estimativa de 0,17% provém dessa única divergência. Explicitar essas contagens, para que o leitor não suponha uma comparação distribuída por muitos subgrupos independentes.
3. **Evento dentro do ciclo:** especificar em prosa ou pseudocódigo que a política decide a quantidade usando o estado anterior à chegada; depois chega o pedido mais antigo do pipeline, o novo pedido é acrescentado, a demanda é atendida com perda de excesso, e manutenção é cobrada sobre o estoque remanescente. Nessa indexação, o pedido feito em t chega em t+3 para o parâmetro L=2. A equação de custo com estoque final é compatível; não descrever chegada t+2.
4. **Reprodutibilidade do método:** adicionar apêndice ou suplemento com limiares e precedência dos perfis, fórmulas/parâmetros das cinco heurísticas, regra de desempate e fallback, convenção de desvio padrão populacional usada na inicialização e código congelado. Os nomes dos algoritmos, isoladamente, não especificam o experimento. VendorResponsive é uma formulação inspirada em descrição textual, não reprodução exata de um algoritmo plenamente especificado por Zabraoui.

Essas solicitações foram enviadas diretamente ao agente redator e ao principal; não modifiquei o LaTeX para evitar conflito de autoria de arquivos.

## Limites científicos que permanecem para submissão

1. O estudo temporal cobre uma única origem e onze ciclos de teste, com resets e custos ilustrativos. Sensibilidade de estoque inicial, pipeline, horizonte e crédito terminal e múltiplas origens seriam a próxima evidência útil. Sem isso, manter estritamente a conclusão sobre esse diagnóstico, sem recomendar uma regra universal.
2. O caso atual não responde se outros perfis, um metamodelo supervisionado ou um portfólio com EOQPerCycle teriam valor. A sensibilidade demonstra fragilidade do comparador, não reavalia o seletor com um portfólio inteiramente saneado. Não estender a conclusão negativa a toda seleção contextual.
3. Os controladores históricos não foram regenerados sob um snapshot identificável. As etiquetas RL/GA não devem ser usadas para alegações gerais de superioridade/degeneração dos algoritmos nem para comparação de eficiência computacional. O artigo já se apresenta como auditoria condicional.
4. As lojas compartilham nove produtos e condições comerciais. O bootstrap por loja preserva dependência interna de loja, mas não resolve dependência cruzada por produto nem o fato de seletores de dobras distintas terem treinamento sobreposto. Os intervalos condicionais são exploratórios.
5. A inclusão de séries deriva de uma coorte previamente preparada e pode ter usado informação futura. A nova separação de calibração não desfaz esse viés de amostragem. Uma replicação com critérios de inclusão restritos ao passado ou dados públicos é necessária para alegações externas fortes.
6. Autorizações dos dados, disponibilidade pública de artefatos permitidos, autoria, financiamento e conflitos de interesse ainda precisam de validação humana. A ausência dessas confirmações impede chamar o pacote de pronto para submissão, embora não impeça o rascunho técnico.

## Julgamento final das alegações

Sustentadas: neste caso e nestas regras, o perfil não mostra redução de custo em relação ao uniforme no limiar principal; a vantagem aparente contra EOQ histórico depende do ponto de serviço e da implementação; serviço médio não garante serviço local; correção dimensional do baseline melhora os dois objetivos agregados no diagnóstico temporal.

Não sustentadas: superioridade universal do EOQPerCycle; inutilidade de todos os perfis ou seletores; ganho financeiro real; validação nacional ou em produção; treinamento independente repetido cinco vezes; reprodução comprovada dos algoritmos do artigo-base.

## Adendo após as correções finais

Inspecionei novamente o LaTeX, o script e o manifesto temporal após as alterações finais. Os hashes do script e do arquivo de cenários conferem com `results.json`. O agente principal informa compilação bem-sucedida do PDF; esta revisão científica não substitui a inspeção visual do arquivo.

**As quatro solicitações textuais acima foram resolvidas:** High Volatility Seasonal está padronizado; a diferença temporal em apenas uma série e as escolhas 144/1 aparecem explicitamente; o Apêndice A detalha precedência e limiares dos perfis, fórmulas e parâmetros das heurísticas, desvio populacional nas políticas versus desvio amostral nas features, desempates/fallbacks e evento pré-chegada com pipeline L+1. O texto esclarece também que CV² inclui ciclos zero, divergindo da variância de tamanhos positivos da classificação original. As regras publicadas são consistentes com o código inspecionado para a calibração de 17 ciclos. A sazonalidade não estimável e as interpretações operacionais de FixedInterval/VendorResponsive são reconhecidas.

A preocupação sobre não reajustar o seletor com o baseline corrigido também recebeu uma análise adicional, declarada como emenda de auditoria posterior. `expanded_summary.csv` e `expanded_selections.csv` registram o portfólio de seis políticas. No limiar principal 0,70:

| Seletor com seis políticas | Custo médio | Serviço médio | Serviço ponderado | Séries abaixo de 0,70 |
|---|---:|---:|---:|---:|
| Global | 167,058674 | 0,895606 | 0,828307 | 19 |
| Perfil | 167,590096 | 0,892873 | 0,825132 | 20 |

O global escolhe EOQPerCycle nas 145 séries; o perfil escolhe EOQPerCycle em 144 e Newsvendor em uma. Portanto o portfólio expandido também não mostra vantagem agregada de custo ou serviço para o perfil no limiar principal. O custo perfil é aproximadamente 0,3181% maior. Essa análise sana a omissão específica do portfólio corrigido, mas continua sendo uma sensibilidade pós-auditoria, não uma replicação independente ou um resultado pré-especificado. Não foram criados novos intervalos bootstrap para esse contraste; não atribuir a ele os intervalos do portfólio de cinco políticas.

**Situação final:** não identifico inconsistência numérica central ou alegação de superioridade incompatível com os resultados na versão inspecionada. Mantém-se a decisão de rascunho defensável, com revisão maior antes de submissão. Os obstáculos substantivos restantes são horizonte/origem temporal únicos e curtos; resets, estoques e custos ainda sem robustez econômica; dependência entre lojas por produto e incerteza condicional; coorte retrospectivamente selecionada; ausência de regeneração integral dos controladores históricos; e confirmação humana de autoria, dados, declarações e suficiência da contribuição para a revista. Resultados negativos foram preservados e seu alcance foi delimitado corretamente.
