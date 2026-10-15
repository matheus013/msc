# Rodada corrigida autorizada em 14/10/2026

O usuário autorizou repetir Bahia e M5 e, em seguida, avaliar todas as séries elegíveis da base nacional. Esta rodada não aplica limite nem amostragem de séries. Os resultados históricos e os PDFs são preservados.

## Protocolo e escopo

- Bahia: partição bruta `data/source/vendas/uf=BA`, agregação por revendedor/produto/ciclo sem eliminar vendas complementares. Origens 17 e 23; 53.396 e 88.865 séries elegíveis, respectivamente.
- M5: todas as séries FOODS, 92 ciclos completos de 21 dias, sem suavização por IQR ou filtro de CV calculado no futuro. Origens 50/64/78; 11.780/13.231/14.192 séries elegíveis.
- Brasil: todas as séries elegíveis em cada UF identificada disponível. Há 26 UFs; a partição de RO está ausente. Existe também uma partição sem UF, que não pode ser atribuída a um estado sem informação adicional. Portanto, a execução nacional não permite afirmar cobertura integral das 27 UFs.
- Elegibilidade: pelo menos três ciclos positivos no prefixo de treinamento. Histórias com zero, um ou dois ciclos positivos são contabilizadas separadamente. Séries maduras que ficam sem demanda no teste continuam na avaliação; seu nível de serviço é indefinido, não artificialmente zero ou um.
- Calendário real comum: 202301–202504, com 17 ciclos por ano completo. Estados com registros mais recentes conservam os ciclos anteriores como posições sem vendas registradas, em vez de deslocar suas janelas. A hipótese de que ausência de eventos significa zero vendas, e não falha de cobertura da fonte, ainda exige validação da origem dos dados.
- Validação e teste: sete ciclos cada, parâmetros e perfil determinados somente pelo prefixo. Estoque inicial e pedidos em trânsito comuns entre candidatos, reconstruídos pela referência EOQ com histórico conhecido; entrega exatamente no lead time de dois ciclos.
- Custos: posse 1, ruptura 5, pedido fixo 50, pedido variável 0,5; premissas de pesquisa, ainda sem validação empresarial. Sem crédito de salvamento terminal; estoque comprometido terminal é exportado.

## Políticas e orçamento

São 12 políticas: EOQCycle, sSCycle, NewsvendorHeuristic, FixedInterval, VendorResponsive, CappedBaseStock, BigDataNewsvendor, GA, SA, DQN, PPO e GA-DQN. O EOQ e os limites de busca usam demanda por ciclo; limites GA/SA são dimensionados somente pelo treinamento, sem pisos absolutos de 2.000 unidades para séries pequenas.

GA: população 100, 50 gerações. SA: 32 cadeias, 156 iterações. Os dois usam a mesma avaliação com penalidade de déficit de serviço; a penalidade não garante viabilidade de uma restrição rígida. DQN e PPO: pelo menos 20.000 transições de ambiente por treinamento, arredondadas para episódios completos, independentemente da base e da origem. GA-DQN repete a busca GA, adiciona 1.000 transições demonstrativas ao replay e depois treina DQN com o mesmo orçamento; seu custo computacional adicional deve entrar na comparação.

Cada método estocástico é treinado de novo com sementes 11, 29 e 47. São 22 calibrações e 44 registros de avaliação por caso de série/origem. Sementes não constituem novas lojas independentes. O perfil mantém a definição legada de CV² sobre todos os ciclos como comparador explícito; esta rodada não valida a definição clássica de variância dos tamanhos positivos. Previsão e PSE supervisionado não são contribuições testadas por este executor.

## Escala e execução

Só Bahia e M5 totalizam **181.464 casos de série/origem e 3.992.208 calibrações**. No teste sintético com orçamento completo, um treinamento DQN levou aproximadamente 87 segundos nesta máquina. Aplicar esse tempo aos 544.392 treinamentos DQN planejados para Bahia e M5, sozinho, corresponde a aproximadamente 1,5 ano de execução serial. Essa é uma extrapolação de custo, não um prazo previsto: históricos reais, comprimento das janelas e outras políticas alteram o tempo. O Brasil adiciona volume a essa fila. A opção censitária por rede individual é computacionalmente muito cara; avaliações futuras devem considerar treinamento compartilhado ou vetorização de várias redes, com protocolo e testes próprios, antes de prometer uma conclusão rápida.

O executor usa CUDA quando disponível e não abre vários processos competindo pela mesma GPU. Bahia e M5 alternam casos; depois a fila percorre as UFs nacionais disponíveis. Bahia é reaproveitada na etapa Brasil, pois dados, origens, configurações e sementes coincidem, com a proveniência registrada e sem contá-la como novo treinamento independente.

A preparação nacional completa encontrou **1.625.351 casos de série/origem**. Somada ao M5, a fila representa **36.620.188 calibrações únicas**, com Bahia reutilizada na composição Brasil. AP, MS, MT, RR e TO têm partições locais, mas nenhum caso elegível nestas origens; RO não tem partição. Esses números mostram que o protocolo com treinamento individual de todas as redes é impraticável como prazo curto de produção de um artigo, mesmo com CUDA. A fila registra o pedido censitário; sua execução não deve ser confundida com garantia de conclusão em prazo útil.

```powershell
& experiment/.venv/Scripts/python.exe experiment/run.py research-production --run census_2026-10-14
```

Para iniciar/retomar em segundo plano, sem abrir janela:

```powershell
& experiment/scripts/operations/start_research_production.ps1 -Run census_2026-10-14
```

Os artefatos locais ficam em `experiment/data/08_reporting/research_v2/census_2026-10-14/`:

- `status.json`: PID, tarefa atual e contagem de calibrações concluídas;
- `manifest.json`: protocolo, coortes, hashes das fontes e do código;
- `results.sqlite`: resultados de cada treinamento, gravados atomicamente após validação e teste;
- `panels/`: painéis elegíveis por UF/origem preparados em disco;
- `code_snapshot/`: código que identifica a rodada;
- `stdout_*.log` e `stderr_*.log`: acompanhamento do processo em segundo plano;
- `kpis.csv`: exportação após Bahia/M5 e depois da etapa nacional.

Consulta do andamento, sem iniciar treinamento:

```powershell
& experiment/.venv/Scripts/python.exe experiment/run.py research-status --run census_2026-10-14
```

A retomada rejeita código, dados ou painéis alterados. Uma calibração interrompida é reiniciada com sua semente; resultados já concluídos não são refeitos. Exceções interrompem a fila e ficam registradas, em vez de gerar silenciosamente uma tabela incompleta. Suspensão ou desligamento da máquina interrompem a execução; o mesmo comando retoma os checkpoints. Não há agendamento de inicialização automática do Windows nem notificações externas.

Verificação: 27 testes passaram (22 do protocolo e integração; cinco da fila), incluindo causalidade, entrega exata, equivalência CPU/CUDA, preservação de vendas complementares, limites de busca, calendário nacional, transação interrompida, retomada do SQLite e conflito de leitura de arquivos no Windows. A verificação sintética com orçamento completo terminou com 22 calibrações e 44 avaliações, cobrindo as 12 políticas e suas sementes; é separada da evidência experimental real.

Na inicialização real, o Windows bloqueou temporariamente a substituição de `status.json` durante uma leitura. Nove calibrações já gravadas foram preservadas. A correção acrescenta tentativas de escrita e um arquivo alternativo `status_latest_<PID>.json` para telemetria, sem permitir que esse bloqueio invalide uma calibração. A migração foi verificada por comparação de AST: nenhuma instrução fora de `write_json` mudou, e os demais hashes científicos permaneceram idênticos. `manifest_before_status_io_repair.json` e `code_snapshot_before_status_io_repair/` preservam a proveniência anterior. A fila foi retomada e avançou para 13 calibrações, com DQN treinando na Bahia; M5 permanece na alternância seguinte e Brasil na etapa posterior.
