# Reescrita dos capítulos técnicos

Reescritos integralmente `capitulos/conceitos.tex` e `capitulos/proposta.tex`, mantendo a estrutura de capítulos, as equações centrais, seus identificadores e as referências bibliográficas usadas na argumentação. Não houve alteração de fonte, margem, espaçamento ou deslocamento de prosa para os apêndices.

As contagens locais, calculadas com uma mesma expressão regular antes/depois, são 5.577 → 2.635 unidades em conceitos (−52,8%) e 11.176 → 5.286 em metodologia (−52,7%). Por espaços, são 4.878 → 2.236 e 9.873 → 4.766. Esses indicadores incluem comandos e notação LaTeX; a verificação de páginas compiladas é a medida final da redução. Os números e a lista de labels estão em `methods_counts.json`.

## Cortes realizados

- Removidas introduções de seção que repetiam a arquitetura, recapitulações do papel das métricas e descrições duplicadas de estado, ação, ambiente, rótulo e PSE.
- Unificados descrição da simulação, ordem dos eventos, validação temporal e unidade experimental.
- Substituída narrativa de módulos, bibliotecas e artefatos por descrição científica de entradas, pressupostos e operações; os parâmetros detalhados continuam referenciados no apêndice existente.
- Removidos quatro diagramas que repetiam uma tabela ou a arquitetura principal: GA--RL, quadrantes ADI/CV², fluxo decisório AIPE e diagrama ASP de Rice. Retidos a tabela de quadrantes, a arquitetura central do AIPE e o esquema temporal.
- Removida a tabela de classificação qualitativa de métodos MEIO, sem evidência comparativa computacional que sustentasse sua ordenação.
- Encurtadas explicações de algoritmos sem eliminar a distinção entre regra de decisão, busca de parâmetros e política aprendida.

## Cautelas científicas incorporadas

- Cinco repetições determinísticas não equivalem a cinco treinamentos independentes ou aumento de poder por common random numbers.
- PSE é componente proposto; arquivos de ajuste e acurácia de treino não demonstram validação nacional ou implantação.
- A fila atual entrega pedido em t+L+1; a dinâmica de vendas perdidas e a cobrança pós-demanda foram explicitadas.
- O lote EOQ legado acumula demanda no horizonte com manutenção por ciclo; não foi apresentado como fórmula canônica dimensionalmente consistente.
- CV² de tamanhos positivos foi separado do CV² do código, que inclui zeros; índices nulos/constantes e janelas sazonais insuficientes receberam ressalvas.
- A hierarquia empresarial não foi tratada como otimização conjunta de múltiplos escalões: o simulador opera séries separadas.
- Foram removidas garantias de ausência de vazamento, causalidades não testadas sobre degeneração RL e afirmações de inexistência de literatura.
- A existência de resultados M5 é reconhecida, com configuração e validação distintas da Bahia; não foi mantida a alegação de ausência de resultados dessa base.

## Labels removidos

`fig:ga_rl_arch`, `fig:sb_quadrant`, `fig:aipe_decision_flow`, `fig:asp_rice`, `tab:met_comp_proposta`. A busca nos capítulos, figuras e diagramas ativos identificava suas referências somente nos próprios trechos removidos. Não foram criados destinos fictícios nem transferidas as figuras para outro capítulo. Fontes e PDFs existentes permanecem preservados no arquivo original.

Todos os labels de equações e seções existentes foram mantidos. Foi eliminada a definição duplicada de `sec:selecao_contextual`. Nenhuma referência ao apêndice de representações ilustrativas foi introduzida. A compilação global e a revisão de referencias cruzadas dependem da consolidação dos outros capítulos pelo agente principal.
