# Protocolo complementar fixado antes da execução (14/10/2026)

Pergunta: perfis calculados no passado ajudam a selecionar uma heurística para ciclos futuros, em comparação com a melhor heurística global escolhida no mesmo conjunto de validação?

- Fonte: `experiment/data/03_primary/scenarios.parquet`, sem alterar o original.
- Universo: séries existentes com pelo menos 33 ciclos; nenhum filtro por desempenho futuro.
- Calibração: primeiros 17 ciclos; validação: próximos 10; teste: restante.
- Portfólio fixo de cinco heurísticas existentes: EOQ, sS, Newsvendor, FixedInterval e VendorResponsive. Não é uma repetição do benchmark de 12 políticas nem comparação de treinamento de RL.
- Todas as estatísticas e os perfis são calculados exclusivamente nos 17 ciclos de calibração. Reusar regras de perfil existentes; ignorar a coluna de política recomendada pelo mapeamento heurístico.
- Custos de simulação: manutenção 1/unidade/ciclo, falta 5/unidade, pedido fixo 50 e variável 0,5/unidade; z=1,28. São unidades de objetivo, sem alegar calibração monetária empresarial.
- Prazo: manter a dinâmica exata do simulador, cujo pipeline de tamanho L+1 entrega pedido t no ciclo t+L+1 para L=2. Explicitar esta convenção; não chamá-la de prazo efetivo de dois ciclos.
- Estoque inicial: média*2 + 1,28*desvio*sqrt(2), calculado apenas na calibração. Pipeline inicial vazio. Validação e teste são episódios separados, reiniciados com esse mesmo estoque. Sem crédito pelo estoque/pipeline terminal. Limitação explícita: episódios curtos e efeito de borda.
- Para cada loja-alvo, selecionar políticas somente pelas validações das outras lojas. Critério: menor custo médio com fill rate médio >= alpha; se inviável, maior fill rate e desempate por custo. Se perfil ausente nas demais lojas, usar seletor global.
- Alpha principal 0,70; sensibilidades predefinidas 0,80/0,85/0,90/0,95. Não escolher limiar com base no teste.
- Avaliar no teste custo total, fill rate médio por série, demanda ponderada atendida e séries abaixo do limiar. Comparar perfil vs global e EOQ, sem atribuir redução de custo a melhoria pura quando serviço cair.
- Bootstrap pareado de 5.000 amostras por loja, semente 20261014; intervalos condicionais às políticas selecionadas, sem refazer seleção. Não são replicações de treinamento ou inferência populacional forte. Dependência entre lojas por produto permanece.
- Heurísticas escalares pequenas em CPU; nenhuma operação de treinamento neural neste experimento justifica custo de GPU. O projeto mantém CUDA disponível para experimentos com redes/otimização vetorial.

Protocolo definido após a auditoria dos resultados históricos; não é um estudo prospectivo pré-registrado. Não ajustar regras/portfólio após ver o teste para recuperar uma hipótese positiva.

## Emenda da auditoria, antes da segunda execução

O revisor identificou uma série sem demanda no teste: manter seu custo e convencionar fill rate=1 quando não há demanda; também informar a média excluindo casos sem demanda. A implementação original retorna 0 nesse caso. Não foi alterado o simulador histórico.

O lote da implementação EOQ histórica usa demanda total de 17 ciclos com manutenção por ciclo. Acrescentar, como sensibilidade de correção dimensional, a mesma política de ponto de reposição com Q=sqrt(2*mu*K/h), chamada EOQPerCycle. Mantê-la fora da escolha original de cinco políticas, mas compará-la explicitamente aos seletores no teste, incluindo contraste bootstrap. Não chamar esta alteração de pré-registrada: surgiu da auditoria após a primeira execução. É correção de unidade, sem ajuste aos resultados futuros. Se dominar seletores em custo e serviço, reportar essa evidência claramente.
