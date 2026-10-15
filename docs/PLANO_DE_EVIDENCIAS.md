# Perguntas de pesquisa e evidências

O experimento deve produzir uma matriz coerente de evidências. Versões anteriores ficam preservadas para rastreabilidade, sem compor um segundo conjunto de resultados na argumentação. Não se concatenam resultados com diferentes definições de custo, preparação ou avaliação.

## Artigo

**Pergunta central:** selecionar a política por perfil melhora o compromisso custo–atendimento frente à seleção global com a mesma informação passada?

| Análise necessária | Evidência | Situação |
|---|---|---|
| Qualidade das candidatas | Custo, atendimento médio e ponderado; fronteira descritiva | Bahia completa; M5 em execução |
| Valor incremental do perfil | Comparação pareada global × perfil em teste reservado; violações individuais | Bahia completa; M5 pendente |
| Estabilidade da recomendação | Escolhas por perfil e origem; distribuição de atendimento | Bahia completa; M5 pendente |
| Dependência da meta | Limiares 0,70 / 0,80 / 0,85 / 0,90 / 0,95; contingências explícitas | Bahia disponível; M5 pendente |
| Incerteza | Reamostragem por loja com origens preservadas; limites condicionais explícitos | Bahia disponível; M5 pendente |

As duas bases são estudos separados que respondem à mesma pergunta. O artigo não precisa reunir todos os módulos, pilotos ou resultados da dissertação. O M5 só entra nas conclusões quando sua matriz estiver suficientemente completa; o andamento não é evidência final. Na Bahia, há 145 séries, duas origens e 4.350 ajustes completos.

## Dissertação

A dissertação comporta a formulação ampla do AIPE, descrição dos dados, justificativa do portfólio, implementação, custo computacional e análises complementares. A síntese experimental deve adotar o mesmo protocolo explícito e indicar para cada pergunta o que foi medido e o que permanece em aberto.

Não estão demonstrados apenas pela execução atual: eficácia do PSE supervisionado; economia monetária real; efeito causal da hibridização com esforço computacional equivalente; robustez a outros custos e prazos; desempenho nacional. Cada afirmação exigiria uma análise ou execução própria. Essas extensões não são requisitos automáticos do artigo.

## Critério de fechamento

Primeiro consolidar as matrizes Bahia e M5, sem repetir ajustes compatíveis. Depois gerar as mesmas análises por base, conferir denominadores, cobertura e proveniência e revisar as conclusões. Não eliminar políticas somente porque perderam no teste; qualquer redução definitiva do portfólio deve considerar literatura, redundância, custo computacional e resultados nas duas bases.

## Figuras do artigo

1. Fluxo experimental: quem usa treino, validação e teste.
2. Custo × atendimento: diferença entre preço baixo e compromisso competitivo.
3. Distribuição do atendimento: onde a média esconde insuficiência individual.
4. Escolhas por perfil e origem: onde a contextualização altera a decisão.

As figuras são geradas por `experiment/scripts/documents/article_research_figures.py` a partir dos registros completos e das escolhas auditadas na validação. Não executam treinamento nem selecionam políticas usando o teste.
