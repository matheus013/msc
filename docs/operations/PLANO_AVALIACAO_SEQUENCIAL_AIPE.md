# Planejamento da avaliação sequencial do AIPE

16/10/2026. **Estado científico: NOT_READY.** Este documento e o planejador organizam uma futura avaliação. Não foram treinadas políticas, produzidas novas recomendações, simuladas trajetórias ou alterada a fila Bahia/M5. A execução existente continua avaliando seleção por bloco.

## O que foi preparado

O planejador `experiment/scripts/operations/plan_aipe_sequential.py` lê exclusivamente os planos e os índices/cabeçalhos dos painéis congelados. Confere os hashes dos painéis, identifica cada série e produz agenda e manifesto em `experiment/data/audits/sequential_plan_2026-10-16/{BA,M5}/`. Não consulta o SQLite ativo, importa módulos científicos ou utiliza métricas de teste. A leitura Parquet desativa threads; não há treinamento ou paralelismo experimental.

A configuração está em `experiment/conf/aipe_sequential_protocol_v1.json`. A grade inicial é uma proposta para planejamento, não uma escolha confirmatória aprovada:

| Dimensão | Opções da agenda |
|---|---|
| Período básico de observação/ação | Ciclo comercial Bahia; bloco de 21 dias M5 |
| Revisão do PSE | A cada 1 ou 2 períodos nativos |
| Horizonte de planejamento | 3 ou 7 períodos nativos |
| Histórico admissível | Expansivo ou janela fixa de 12 períodos |
| Latência dos registros | Zero períodos completos, como hipótese explícita não verificada |
| Atualização do seletor | Artefato inicial congelado por padrão |
| Recalibração dos controladores | Artefatos iniciais congelados por padrão |
| Acompanhamento | Os mesmos 7 períodos de teste por episódio |

São oito cenários por caso. Bahia: 290 casos série–origem e 12.760 linhas de decisão. M5: 5.326 casos e 234.344 linhas. As quantidades de linhas não são tamanho amostral independente, número de ajustes nem progresso de execução. As coortes históricas permanecem as mesmas: não representam toda Bahia ou todas FOODS. Não se deve chamá-las de reserva confirmatória sem auditoria própria.

## Relógio e informação disponível

Posições são índices começando em zero, com fins de intervalo exclusivos. Na origem `o`, o treino original ocupa `[0,o)` e a validação ocupa `[o,o+7)`. A primeira revisão ocorre em `t=o+7`, antes da demanda do primeiro período de teste; o episódio termina em `o+14`. Na decisão `t`, somente registros em posições menores que `t−latência` estão admissíveis. Uma janela fixa seleciona os últimos períodos desse prefixo; a expansiva começa em zero.

Esse prefixo define um **limite de informação admissível**, não autorização para ajustar usando qualquer rótulo presente. Artefatos iniciais precisam respeitar a validação e as partições por loja estabelecidas. Resultados-alvo ainda não observados e rótulos de desempenho cujo horizonte não terminou nunca podem entrar em treino. O calendário próprio de atualização, quando habilitado, também exige maturação de rótulos e proveniência do conjunto usado. O planejador não implementa esse treino nem resolve sua disponibilidade.

A cadência não altera o período básico de ação: o estoque continua evoluindo a cada período nativo. A recomendação permanece válida até a próxima revisão ou o fim do episódio. Por exemplo, com revisão a cada dois períodos, as posições do teste são `o+7`, `o+9`, `o+11`, `o+13`; a última recomendação vigora por apenas um período observado. Reavaliar pode resultar em manter a mesma família.

O horizonte de planejamento pode atravessar revisões. O campo `planning_end_exclusive` registra o horizonte solicitado, inclusive quando ultrapassa o fim do acompanhamento. O campo `scorable_horizon_end_exclusive` limita os resultados realizáveis aos sete períodos acompanhados; as duas extensões não são confundidas. Um futuro executor precisará de previsões/condições terminais declaradas para planejar além desse limite. Nenhuma observação futura é disponibilizada ao modelo por essa agenda.

Atualização de seletor e recalibração de controladores têm relógios separados do PSE: intervalo zero mantém os artefatos iniciais; um inteiro positivo agenda atualizações depois desse número de períodos, mesmo quando não coincide com revisão. As posições aparecem no manifesto e nas linhas da agenda. São eventos planejados, não atualizações realizadas. O que fazer com parâmetros atualizados durante a vigência de uma recomendação também deverá ser congelado no executor, sem troca silenciosa da família.

## Estado e custos

Cada combinação caso–cenário–estratégia–semente terá trajetória própria. No início de um episódio, as estratégias recebem condições iniciais comparáveis. Nas revisões seguintes devem carregar saldo, pedidos em trânsito, compromissos e memória da política. A mudança de família exige um adaptador de estado que não cancele pedidos nem reinicie o aquecimento EOQ. Origens distintas continuam episódios separados; não se alega continuidade entre elas. Os testes Bahia se sobrepõem em um ciclo e compartilham séries, logo não são independentes.

Custos de cenário mantidos: `h=1` por unidade em saldo ao fim do período nativo; `p=5` por unidade não atendida; `K=50` por pedido positivo; `c=0,5` por unidade pedida. Prazo exato de dois períodos nativos e alvo nominal de fill rate de 0,70. Não são reais/dólares calibrados. Mudar somente a cadência de revisão não multiplica `h`, `K` ou os demais custos, nem converte o prazo em número de revisões. No M5 são 42 dias de prazo; Bahia permanece em dois ciclos comerciais, sem conversão civil presumida.

Alterar a resolução de observação/ação exige outro desenho: reagregação dos registros admissíveis, calendário, transformação correta das taxas, prazos e restrições, além de recalibração. O planejador rejeita resolução diária e frações de período. A existência de vendas diárias no acervo M5 não significa que esta agenda ou a fila atual valide ações diárias. Para Bahia, não se inventa resolução mais fina a partir dos agregados.

## Por que a execução ainda está bloqueada

Os KPIs agregados dos sete períodos não permitem reconstruir decisões sequenciais. A auditoria identificou ausência de serialização completa das políticas/seletores, parâmetros finais, trajetórias e custos desagregados no caminho canônico. Nenhum custo agregado foi redistribuído entre ciclos para simular uma trajetória inexistente.

Antes de executar é necessário:

1. Persistir políticas ajustadas, parâmetros finais, redes e seletores com versões, hashes e dados de treino.
2. Registrar estados iniciais e transições por período: saldo, trânsito, pedidos, recebimentos, demanda, atendimento e componentes de custo.
3. Implementar e verificar continuidade de estado e adaptação entre famílias, incluindo memória de controladores e efeitos de mudança.
4. Emitir recomendação, alternativas, previsões e justificativa antes de revelar a demanda-alvo; pontuar depois em registro separado.
5. Garantir disponibilidade temporal dos dados e maturação dos rótulos em qualquer atualização; declarar o que é hipótese de latência.
6. Congelar referência, estratégia primária, regra de recuo, partições, orçamento e auditoria da reserva. A grade proposta não supre essas escolhas.
7. Resolver a semântica das lacunas de vendas Bahia e manter custos/estoques como cenários enquanto não houver calibração operacional válida.

Ausência de artefato não será suprida apresentando um novo ajuste como recuperação exata de um checkpoint antigo. Quando necessário, novos ajustes/replays deverão ser versionados como execução adicional, fora da fila congelada.

## Reprodução e verificações

Na raiz do projeto:

```powershell
experiment/.venv/Scripts/python.exe -m unittest experiment/tests/test_aipe_sequential_plan.py
experiment/.venv/Scripts/python.exe experiment/scripts/operations/plan_aipe_sequential.py --dataset both
```

Para outra grade, copie a configuração e passe `--config CAMINHO_JSON`; a saída permanece restrita à pasta de auditoria indicada. O comando sobrescreve apenas os arquivos de planejamento dessa pasta, nunca resultados científicos. Use `--dataset BA` ou `--dataset M5` para planejar uma base. Os manifestos incluem hashes do script, configuração, planos e painéis de origem, contagens e dependências; todos mantêm `scientific_execution_status: NOT_READY`.

Nove testes verificam exclusão de informação futura, inclusive com uma pequena série-fixture contaminada no futuro; janela fixa e expansiva; latência; bordas de cadência/horizonte; falta de histórico; resolução incompatível; requisitos de estado e artefatos; relógios de atualização separados; e preservação da unidade do custo fixo. A verificação integrada percorre ambas as agendas geradas e confere hashes, contagens, limites temporais e inicialização única por caso/cenário; é ignorada quando os artefatos ainda não foram gerados. Esses testes validam o planejador, não a dinâmica do futuro simulador.

## O que mudaria nas alegações do estudo

A implementação e execução futuras permitirão avaliar revisão contextual periódica com estado persistente e comparar cadências específicas. Até lá, os textos devem distinguir o requisito metodológico parametrizável da evidência atual de escolha por bloco. Não há demonstração de economia em reais, adoção gerencial, validade para qualquer cadência ou superioridade do AIPE nesta nova configuração.
