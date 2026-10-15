# Diagnóstico e extensão conservadora do PSE — M5

Análise de desenvolvimento do piloto de 30 séries e 90 casos, realizada em
15/10/2026. Não foram executadas novas simulações de estoque. Foram reproduzidas
exatamente as 450 recomendações originais, preservando o experimento e seu
snapshot. O código do seletor foi refatorado para expor suporte das classes,
escores e importância das características, sem alterar essas recomendações.

## O que foi identificado

- O classificador reproduz 100% dos rótulos do próprio treino, mas apenas 19/90
  rótulos das lojas excluídas. Cada fold tem 27 exemplos para até sete classes;
  dois rótulos avaliados pertencem a classes ausentes no treinamento.
- A vencedora da validação permanece vencedora no teste em 29/90 casos. Aceitando
  empates numéricos no objetivo completo, são 33/90. Com demanda positiva nos
  dois blocos, a coincidência nominal é 26/83. Trocas de nome não são, sozinhas,
  evidência de perda operacional relevante.
- As vencedoras variam entre as três sementes em 39/90 casos. Com apenas três
  sementes, isso caracteriza sensibilidade, sem estimar uma probabilidade precisa.
- O perfil explícito tem importância por ganho igual a zero em todos os folds;
  com e sem essa entrada, as 90 recomendações coincidem. Ele é derivado das
  mesmas características numéricas já fornecidas ao classificador.
- As três origens têm as mesmas 26 séries esparsas e quatro de tendência.
  A concentração existe também na coorte elegível: 826/885, 1.977/2.033 e
  2.360/2.408 séries esparsas nas origens 50, 64 e 78. High_Vol_Seasonal tem
  somente duas, duas e seis séries, respectivamente. Os rótulos esparsos
  satisfazem a regra explícita, sem uso do perfil padrão para casos não cobertos.

Esses resultados são consistentes com sobreajuste, instabilidade dos rótulos e
pouca diversidade de perfis. Não permitem concluir que aprendizado ou perfis
sejam inúteis em uma população maior ou em Bahia.

## Uma extensão testada sem novas simulações

O PSE preditivo recorre à associação por perfil quando o maior escore de classe
é inferior a 0,60 ou a classe prevista tem menos de três exemplos no treinamento.
Esses limiares foram fixados para uma única extensão conservadora; não foi feita
busca de limiares no teste. O escore não representa probabilidade calibrada de
sucesso operacional. A decisão de recuar usa apenas informações de treinamento
e validação das outras lojas. Houve recuo em 62/90 casos, com alteração efetiva
de política em 48/90.

| Estratégia | Custo médio | Fill rate médio | Casos com serviço ≥70% |
|---|---:|---:|---:|
| Seleção global | 1.086,97 | 71,46% | 47/86 |
| PSE por perfil | 1.094,02 | 70,91% | 46/86 |
| PSE preditivo original | 1.238,30 | 67,35% | 39/86 |
| PSE preditivo com recuo | 1.087,26 | 69,96% | 46/86 |
| Escolha individual na validação | 1.185,13 | 68,38% | 44/86 |

Quatro casos têm demanda de teste zero: entram no custo médio, mas seu serviço é
indefinido. Os custos são por horizonte de sete ciclos e não usam conversão
monetária entre bases. Cada caso agrega primeiro as sementes da política.

O recuo reduz custo em 12,20% e aumenta serviço em 2,62 pontos percentuais frente
ao preditivo original. Ainda custa 0,026% mais que a global e perde 1,49 ponto
de serviço. Seu serviço médio permanece abaixo de 70%, apesar da proximidade.
Não há demonstração de superioridade sobre a global ou o PSE por perfil.

O bootstrap exploratório pareado por loja indica diferença de custo do recuo
frente à global de +0,28, com intervalo de 95% entre −39,83 e +39,30; diferença
de serviço de −1,49 ponto, com intervalo entre −2,87 e −0,23 ponto. A inclusão de
zero no intervalo de custo não prova equivalência. São dez lojas, com origens
dependentes e múltiplas comparações, sem avaliação confirmatória.

Os rótulos do classificador impõem serviço por série; global e perfil usam
serviço agregado nos grupos de validação. Portanto, as abordagens também diferem
na regra de decisão. A moda dos mesmos rótulos foi mantida como controle e teve
custo 1.096,65 e serviço 68,85%: não se deve atribuir toda diferença à escolha
do algoritmo de aprendizado.

## Onde intervir

Para o PSE preditivo original, os 90 casos se decompõem em:

- 39 atendem ao mínimo de serviço;
- 21 falham embora exista uma política viável no portfólio: espaço para melhorar
  a seleção;
- 15 falham sem política viável no portfólio, embora o planejamento com informação
  perfeita consiga atender: espaço para melhorar o controle/calibração;
- 11 têm atendimento mínimo fisicamente impossível devido à demanda e às
  condições iniciais antes das primeiras entregas;
- quatro têm demanda zero e serviço indefinido.

Com recuo, os erros de seleção caem de 21 para 14. Os outros limites permanecem.
WI_3 e TX_2 respondem por 65,4% da diferença líquida de custo entre o preditivo
original e a global. O original apresenta custo maior e serviço menor em ambos
os perfis e nas três origens.

## Encaminhamento

Manter as duas abordagens do PSE e a seleção global como referência. O recuo é
uma extensão exploratória do preditivo, não uma terceira proposta já validada.
Antes da rodada final, precisamos de mais exemplos de treinamento por classe,
suporte explícito para perfis raros e validação interna por loja para eventuais
limiares/hiperparâmetros. A concentração natural do M5 impede pressupor que
apenas aumentar o número de séries produzirá muitos perfis diferentes.

O conjunto usado neste diagnóstico passa a ser conjunto de desenvolvimento.
Qualquer confirmação das melhorias precisa de séries reservadas que não tenham
orientado essas decisões. Não foram iniciados M5 completo, Bahia ou Brasil nesta
etapa. Não foram alterados os resultados do artigo nem da dissertação.

## Reprodução e artefatos

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/analysis/diagnose_pse_pilot.py
```

Saídas locais em
`experiment/data/08_reporting/research_v2/m5_pse_pilot30_v2_2026-10-15/diagnostics_v1/`:
relatório completo, cinco figuras em PNG/SVG, predições da extensão antes da
pontuação, importâncias por fold, transição de rótulos, sensibilidade às sementes,
decomposição das falhas, recortes por loja/perfil/origem e intervalos pareados.
O protocolo registra hashes das entradas e do código. **31 testes passaram**,
incluindo limites do recuo, empates e decomposição das falhas.

O cuidado com desenvolvimento e avaliação segue
[Cawley e Talbot (2010)](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
Escores de classificação não equivalem automaticamente a probabilidades
calibradas; ver a [documentação do scikit-learn](https://scikit-learn.org/stable/api/sklearn.calibration.html).
