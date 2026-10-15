# Portfólio candidato após a revisão de literatura

Data: 14/10/2026. A justificativa e as fontes estão na [revisão dirigida](REVISAO_LITERATURA_POLITICAS.md). Este registro substitui as propostas provisórias de cinco métodos e de seis métodos com Newsvendor.

| Papel | Método |
|---|---|
| Referência simples de lote | EOQ adaptado |
| Reposição até nível-alvo com gatilho | (s,S) |
| Referência para vendas perdidas e lead time | CappedBaseStock |
| Busca de parâmetros de reposição | GA |
| Aprendizado por reforço | PPO |
| Hipótese de integração busca/aprendizado | GA-DQN |

**Controle obrigatório:** DQN isolado, na ablação pareada GA / DQN / GA-DQN. Se a comparação usar apenas uma coorte de ablação, ela não fundamentará uma conclusão de superioridade do híbrido em toda a população. Newsvendor, BigDataNewsvendor, FixedInterval, VendorResponsive, SA e os outros métodos implementados ficam fora do comparativo principal.

Não se afirma que EOQ e (s,S) sejam equivalentes, que DQN seja uma duplicata de PPO ou que os excluídos sejam universalmente inferiores. GA e PPO têm respaldo independente da referência de Zabraoui; o híbrido permanece uma hipótese motivada por ela. Não foi feito um ranking bibliométrico de popularidade.

O registro `experiment/src/simulation/research/portfolio.py` define os seis candidatos e o conjunto de ablação. As configurações Kedro foram alinhadas aos nomes equivalentes. O planejamento de ajustes passou a usar a lista efetiva de políticas e sementes, sem a constante antiga de 22 ajustes por caso. São três políticas determinísticas e três estocásticas: 3 + 3×3 = 12 ajustes por caso na avaliação principal.

O corte não resolve sozinho o custo do censo. O GA-DQN precisa de correção e ablações; o treino integral em lotes ainda precisa de implementação e validação. Os experimentos permanecem parados. O artigo atual e as matrizes históricas continuam documentando o portfólio que efetivamente foi avaliado, sem simular que já há resultados novos.
