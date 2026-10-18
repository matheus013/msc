# Módulo opcional de contexto do AIPE

Implementado em 17/10/2026 para avaliação posterior à execução atual. O módulo permanece desligado na configuração entregue. Nenhum arquivo científico da rodada corrente foi alterado e o runner atual não importa o novo pacote.

## Papel na metodologia

O contexto adicional é uma entrada opcional do PSE preditivo. Ele pode conter atributos cadastrais, comerciais, demográficos ou de calendário disponíveis ao gestor, com origem e disponibilidade histórica documentadas. O procedimento permanece: especificar a decisão, representar o contexto, construir evidência, recomendar e avaliar. O novo módulo amplia a representação, sem modificar automaticamente demanda, custos, ambiente ou controladores.

A finalidade é investigar ganho operacional incremental. O simples acesso a mais colunas não estabelece benefício. As fontes existentes da Bahia continuam sem vigência histórica comprovada; por isso suas definições são entregues com `availability_verified: false`. Não são preenchidas retroativamente com a fotografia atual.

## Escolhas do gestor

Em `experiment/conf/aipe_context_extension_v1.json`, cada atributo possui um modo:

| Modo | Comportamento |
|---|---|
| `fixed` | Permanece na variante ampliada e não participa da eliminação automática. Conta no limite total. |
| `candidate` | Participa da seleção; só é acrescentado se melhorar o critério operacional interno. |
| `excluded` | Fica fora da leitura útil, da seleção e dos modelos. |

O gestor pode, por exemplo, fixar idade e deixar segmento e filial para seleção. Isso declara quais entradas deseja preservar; não certifica ganho. Uma variante ampliada com fixos continua sendo comparada à versão básica. A recomendação padrão mantém o PSE básico quando a evidência interna não favorece a ampliação. As variantes sem essa proteção permanecem disponíveis para comparação e escolha explícita.

Comando de configuração, para usar quando desejado:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_extension.py configure --dataset BA --fixed idade --candidates segmento filial
```

`--excluded` retira atributos; `--enable`/`--disable` mudam somente o módulo opcional. Papéis não mencionados ficam como estavam. Uma cópia anterior é preservada em `experiment/data/context_extension/config_history`. Alterar esse registro não reconfigura a simulação corrente nem altera um modelo já ajustado. Novos papéis ou contratos exigem nova versão do estudo/modelo.

## Contrato dos dados

As chaves dos casos são `origin,warehouse,store_id,item_id`. Há dois arquivos externos:

1. **Cortes temporais:** essas chaves, `decision_at` e `validation_available_at`. O segundo instante informa quando os resultados de validação usados como alvos de aprendizagem estavam disponíveis. Ambos usam ISO 8601 com fuso, por exemplo `2024-01-01T00:00:00Z`; o exemplo não define datas do experimento real. Em cada origem, todos os alvos usados no treinamento precisam estar disponíveis antes das recomendações.
2. **Contexto longo:** `warehouse,store_id,item_id,feature,value,available_at`. Cada linha representa uma observação conhecida a partir de `available_at`. `level` define a unidade: `warehouse`, `store` ou `series`; chaves de níveis mais detalhados podem ficar vazias quando não se aplicam. A junção utiliza o último registro conhecido antes ou no instante da decisão. Valores futuros não substituem os conhecidos.

Cada atributo declara `kind` numérico/categórico, descrição, fonte, nível, modo e evidência da disponibilidade (`availability_evidence`). Preencher uma data ou mudar um booleano não comprova a qualidade histórica: o responsável precisa identificar a fonte que sustenta a declaração. Idade requer referência temporal; segmento requer a regra e a janela de apuração; SNAP e eventos precisam ser agregados às datas reais dos ciclos e ao estado apropriado. Calendários futuros só são admissíveis quando já eram conhecidos na decisão.

Valores conflitantes na mesma chave/instante são rejeitados. Não há propagação automática de cadastro atual ao passado. A junção temporal usa `merge_asof`, evitando multiplicar casos por todos os registros históricos. A leitura do CSV é feita em blocos, descarta atributos excluídos e limita o volume retido. Não há atributos de teste, oráculos, identificadores do caso ou rótulos como candidatos.

## Seleção e avaliação

- As 17 características básicas de demanda e o perfil operacional continuam como referência. Elas são obtidas pelo mesmo código congelado da matriz, sem alterá-lo.
- Em cada origem, uma loja/revendedor inteiro fica fora do ajuste externo, incluindo todos os seus produtos. Atributos são escolhidos somente nas demais lojas. Origens futuras não treinam anteriores.
- Dentro desse treino externo, validação agrupada por loja ajusta imputação, categorias e modelos em cada partição. Os resultados da loja externa não orientam a escolha dos atributos.
- A busca é progressiva: testa a inclusão de cada candidato e conserva a melhor inclusão admissível. Ela pode parar sem escolher nenhum. Trata-se de busca limitada, sem garantia do melhor subconjunto global.
- O critério é reduzir a soma dos custos nos mesmos casos, respeitando tolerâncias de queda do atendimento médio e de aumento da fração abaixo do alvo. Padrões conservadores são zero para essas tolerâncias. São critérios de desenvolvimento configuráveis, não margens empresariais universais ou garantias de não inferioridade.
- O relatório também expõe novas falhas individuais e recuperações, pois uma fração agregada estável pode esconder trocas de casos prejudicados.
- Constantes e candidatos excessivamente ausentes são rejeitados no treino. Atributos fixos constantes são mantidos e sinalizados; um fixo sem nenhum valor admissível no treino impede o ajuste.
- Há limites de atributos, avaliações internas, categorias codificadas e registros de contexto. O padrão admite até três atributos adicionais e 16 avaliações de subconjuntos por treino externo. Se o orçamento não permite testar uma etapa inteira, a etapa não escolhe oportunisticamente apenas os primeiros candidatos.

O modelo pode prever classe ou consequências de custo/serviço (`selection.model`). São usadas florestas ExtraTrees com os mesmos parâmetros entre variantes para isolar o efeito do contexto; a versão por classe não se confunde com o XGBoost histórico. A versão por consequências usa a mesma regra de intervenção nas comparações, sem ajustar margens no teste. Os modelos usam uma thread e o comando Windows inicia com prioridade reduzida.

As quatro estratégias são `PSE_base`, `PSE_context_fixed`, `PSE_context_selected` e `PSE_context_guarded`. Esta última usa contexto somente quando a comparação interna com a básica satisfaz o critério declarado. O teste descreve o resultado posterior; ele não reescolhe atributos nem ativa o módulo retroativamente.

## Comandos e isolamento da execução corrente

Agora, é seguro apenas inspecionar o plano:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_extension.py plan --dataset BA
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_extension.py plan --dataset M5
```

Não há agendamento ou ativação automática após a simulação. Depois de concluir as rodadas, a preparação exporta matrizes existentes e características do prefixo, sem ajustar políticas:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_extension.py prepare --dataset BA --output experiment/data/context_extension/BA/prepared_v1
```

Ela confere conclusão da fonte, hashes dos painéis, dependências científicas e estabilidade das entradas. Gera `features.csv`, `validation.csv`, `test.csv`, templates e manifesto. Não infere datas civis de códigos de ciclo sintéticos. Preencher os cortes, fornecer os registros historicamente admissíveis e apontar seus caminhos na configuração são requisitos anteriores à avaliação.

Quando estiver tudo validado e `enabled: true`:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/context_extension.py evaluate --dataset BA --prepared experiment/data/context_extension/BA/prepared_v1 --output experiment/data/context_extension/BA/study_v1
```

O comando recusa avaliação enquanto qualquer uma das rodadas BA/M5 configuradas não estiver concluída. Saídas só podem ocupar um novo subdiretório de `experiment/data/context_extension`; não há sobrescrita dos resultados canônicos ou de estudos anteriores. Nenhum comando treina controladores ou chama simulação de estoque. Só a avaliação ajusta modelos do PSE.

## Artefatos e uso posterior

O estudo persiste protocolo/configuração/hashes, cobertura temporal, atributos fixos e selecionados por partição, candidatos rejeitados, tentativas da busca, escores internos, razões de parada, modelos ajustados, recomendações congeladas **antes da leitura do teste**, consequências por caso, comparações pareadas e resumos. Os dados/modelos ficam fora do Git.

API em `experiment/src/aipe_context`: `assemble_context`, `fit_context_pse` e `evaluate_context_pse`. Um objeto ajustado expõe `recommend`, que monta contexto válido na data da nova decisão e retorna por padrão a estratégia protegida, sem receber resultados posteriores. `enabled=False` usa a referência básica sem consultar fontes adicionais. O método `predict` fornece as quatro variantes para avaliação. Mudança de tipo/nível de um atributo selecionado exige novo ajuste.

O estudo adicional pode reaproveitar as mesmas matrizes porque modifica somente a entrada e o ajuste do seletor. Alterar controladores, previsão interna, dinâmica, custos ou demandas exige nova auditoria de compatibilidade. Resultados já observados são desenvolvimento; o módulo ainda requer avaliação empírica e confirmação independente antes de afirmar benefício.

A preparação confere os hashes congelados das três dependências usadas na reconstrução dos atributos: caracterização da demanda, definição do perfil e lista de entradas do seletor. Divergências nas demais fontes são registradas no manifesto, sem executar novamente essas fontes. A Bahia possui uma revisão anterior em `oracles.py`; ela não altera a reconstrução dos atributos e fica explicitamente registrada. As matrizes reaproveitadas continuam sendo os arquivos existentes, identificados por hash.

## Verificação

`experiment/tests/test_context_extension.py` usa somente dados sintéticos e cobre informação futura, cortes de rótulos, papéis manuais, seleção por custo/serviço, grupos externos, ausência de ganho, limites de orçamento, categorias novas/ausentes, matrizes incompletas, serialização e ordem de leitura do teste. Não usa os bancos ou as matrizes científicas em andamento.
