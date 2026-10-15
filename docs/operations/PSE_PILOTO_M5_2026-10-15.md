# Piloto do PSE no M5

Protocolo de desenvolvimento iniciado em 15/10/2026, antes de retomar o artigo.
Não representa a avaliação confirmatória final do AIPE.

## Escopo e execução

- 30 séries FOODS: três por loja, nas dez lojas do M5.
- Sorteio reprodutível, semente 20261015: uma série em cada terço de volume médio
  do prefixo de 50 ciclos. Não usa demanda de validação ou teste para amostrar.
- Universo de amostragem: 885 séries elegíveis na primeira origem da coorte
  histórica de 2.408 IDs. Não representa todas as séries FOODS nem cold start;
  a coorte histórica também impede tratar o piloto como réplica exata de Zabraoui.
- Origens 50, 64 e 78; sete ciclos de validação e sete de teste em cada origem:
  90 casos série–origem. Agregação mantida em ciclos completos de 21 dias.
- EOQCycle, sSCycle, CappedBaseStock, GA, PPO, GA-DQN e DQN.
  DQN é o controle necessário para avaliar a contribuição do híbrido.
- Três sementes independentes de treinamento (11, 29, 47) para GA/PPO/GA-DQN/DQN;
  uma execução das três políticas determinísticas. São 1.350 ajustes.
- Orçamentos completos de produção: 20 mil transições, arredondadas para episódios
  inteiros; GA com população 100 e 50 gerações. Não há redução oculta de orçamento.
- Oito processos CPU, uma thread por processo, prioridade abaixo do normal.
- 294 ajustes compatíveis importados da rodada interrompida; faltavam 1.056 ao
  iniciar. Estimativa inicial de 45–90 minutos, condicionada à carga da máquina.

Os 9.483 ajustes da rodada grande permanecem preservados. A importação verifica
integridade dos painéis e do snapshot do código, além da equivalência da AST dos
arquivos científicos. A única exceção é a função de calibração da CappedBaseStock,
que passou a respeitar pesos de risco já explicitados na configuração. Nenhum
ajuste antigo dessa política é importado; os outros seis algoritmos não usam essa
função. O runner histórico continua rejeitando mudanças científicas ao retomar.

## Avaliação sem usar o teste para selecionar

O PSE é treinado separadamente em cada origem, deixando uma loja inteira fora
por vez. Features e perfil usam apenas o prefixo de treinamento. Rótulos usam
somente os custos e serviços de validação das outras nove lojas: menor custo
com fill rate de pelo menos 70%; se nenhuma política atende, maior serviço e,
depois, menor custo. Serviço indefinido por demanda zero permanece indefinido;
quando todas as opções têm serviço indefinido, o desempate usa custo.

Imputação, codificação de perfil e codificação de classes são ajustadas dentro
de cada fold. Classes raras não são removidas; classes desconhecidas na loja
avaliada são contabilizadas. O classificador XGBoost tem hiperparâmetros fixos:
200 árvores, profundidade três, taxa 0,05, sem subamostragem, uma thread.
Treinos com apenas uma classe usam um classificador constante.

Comparadores: PSE com perfil (AIPE), PSE sem perfil, seleção global, seleção por
perfil e moda dos rótulos de treinamento. O último controla diferenças entre
o objetivo de classificação por série e a restrição de serviço agregado dos
comparadores global/por perfil. A política de cada série ainda é calibrada no
seu próprio prefixo: trata-se de transferência do seletor entre lojas, não de
uma avaliação de lojas sem histórico.

Predições são persistidas antes da leitura dos resultados de teste pela análise.
Custos e serviços das políticas estocásticas são primeiro promediados entre
sementes. Intervalos exploratórios usam bootstrap pareado de lojas, mantendo
itens e origens do mesmo grupo juntos. Dez lojas oferecem precisão limitada.

## Melhor observado e limite com informação perfeita

1. Melhor política única no teste, por origem: diagnóstico retrospectivo.
2. Oráculo do portfólio por série–origem, baseado nas médias entre sementes do
   teste: mesma regra de custo sob serviço mínimo. Não é um comparador executável.
3. Planejamento MILP com conhecimento de toda a demanda de teste, mesmo estoque
   inicial, pipeline, lead time e custos. Quantidades são contínuas, sem crédito
   terminal. É um ótimo condicional determinístico, não um máximo universal.

O MILP também calcula o maior atendimento fisicamente possível, considerando
faltas inevitáveis antes das primeiras entregas. Se 70% for impossível, primeiro
maximiza atendimento e depois minimiza custo. Somente soluções com otimalidade
comprovada são chamadas de ótimas; timeout e limites do solver ficam registrados.
O cronograma encontrado é reexecutado no simulador para conferir custo e serviço.

As saídas incluem quantidade e percentual de acertos do oráculo do portfólio,
empates numéricos, frequência até 1% e 5% do menor custo, e eficiência de custo
`100 × custo de referência / custo do AIPE`. A eficiência só é calculada quando
ambas as opções satisfazem o mesmo critério de serviço; baixo atendimento não
recebe crédito por ser barato. Referência e custo ambos zero equivalem a 100%.

O ótimo MILP sob restrição de serviço por trajetória é comparado com cada
realização de treinamento que atende à restrição, e não com uma mistura entre
sementes que atende somente em média. As políticas determinísticas são repetidas
apenas na análise pareada de sementes, sem novos ajustes. Essas realizações não
são séries independentes. A razão entre fill rate realizado e máximo fisicamente
possível é registrada separadamente da eficiência de custo.

## Artefatos e retomada

Runner: `experiment/scripts/analysis/run_pse_pilot.py`.

Execução/retomada:

```powershell
& experiment/.venv/Scripts/python.exe -u experiment/scripts/analysis/run_pse_pilot.py
```

Saída: `experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15/`.
Inclui manifest, snapshot do código, casos, features, SQLite transacional,
status, logs, KPIs, predições congeladas, métricas por caso, intervalos por loja,
soluções MILP, métricas por realização e resumo. A retomada rejeita alterações
de código/protocolo e executa apenas ajustes ainda ausentes. Não lançar duas
instâncias: um lock exclusivo impede escrita concorrente.

Testes: 28 verificações passaram, incluindo vazamento entre lojas, retenção de
classes raras, serviço indefinido, empates, restrição de serviço, lead time,
pipeline inicial, calibração de risco e análise completa com dados simulados.

Referências de implementação: [MILP do SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
[validação por grupos](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out)
e [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
