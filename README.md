# Problema de Pesquisa: Otimização do Abastecimento em uma Rede de Lojas com Múltiplos Centros de Estoque

## Descrição Geral

Em uma rede de lojas distribuídas geograficamente, o abastecimento eficiente de mercadorias é um fator crítico para minimizar custos operacionais e garantir a disponibilidade de produtos para os clientes. Essa rede conta com múltiplos centros de estoque, cada um armazenando um conjunto específico de itens em quantidades limitadas. Devido às restrições logísticas e ao alto custo de manutenção de grandes estoques nas lojas, essas operam com um inventário reduzido, suficiente para atender à demanda de curto prazo em cada ciclo de abastecimento.

O ano é dividido em 17 ciclos denominados "rounds", e para garantir a operação eficiente das lojas, é necessário planejar quais produtos devem ser transferidos dos centros de estoque para as lojas, garantindo o atendimento à demanda do próximo round. Além disso, é possível realizar transferências entre centros de estoque, caso necessário, para otimizar a distribuição dos itens. A ordem dos rounds é cíclica, ou seja, após o round 17, o ciclo reinicia no round 1.

O sistema de abastecimento é baseado em transferências periódicas de mercadorias, sendo que cada transporte tem um custo proporcional à distância percorrida, onde a unidade de custo por distância é representada por uma constante `c`. O desafio central é determinar a melhor estratégia de reabastecimento, de forma que o custo total da rede seja minimizado enquanto a demanda das lojas é atendida de maneira confiável ao longo dos rounds.

## Formulação do Problema

### Parâmetros e Conjuntos
- **\( S \)**: Conjunto de centros de estoque.
- **\( I_s \)**: Conjunto de itens armazenados em um centro de estoque \( s \in S \).
- **\( M \)**: Conjunto de lojas.
- **\( I_m \)**: Conjunto de itens disponíveis em uma loja \( m \in M \).
- **\( O_m \)**: Sequência de demanda esperada para cada loja \( m \in M \) ao longo dos rounds.
- **\( d(a, b) \)**: Distância entre os locais \( a \) e \( b \), podendo ser um centro de estoque ou uma loja.
- **\( c \)**: Custo unitário de transporte por unidade de distância.
- **\( R \)**: Conjunto de 17 rounds que compõem o ano, onde o round 17 é seguido pelo round 1.
- **\( C_{a,b}\)**: Capacidade máxima de transporte entre dois locais \(a\) e \(b\).
### Variáveis de Decisão
- **\( x_{a,b,i,r} \)**: Quantidade do item \( i \) transferida do local \( a \) para o local \( b \) no round \( r \).

### Função Objetivo
Minimizar o custo total de transporte:
\[
\min \sum_{a \in S \cup M} \sum_{b \in S \cup M} \sum_{i \in I_a \cap I_b} \sum_{r \in R} c \cdot d(a, b) \cdot x_{a,b,i,r}
\]

### Restrições
1. **Disponibilidade nos Centros de Estoque**
   - A quantidade de itens transferidos de um centro de estoque não pode exceder sua disponibilidade:
   \[
   \sum_{b \in S \cup M} \sum_{r \in R} x_{s,b,i,r} \leq ê_s(i), \quad \forall s \in S, i \in I_s
   \]
   onde \( ê_s(i) \) representa a quantidade inicial do item \( i \) no centro \( s \).

2. **Atendimento da Demanda das Lojas**
   - A quantidade recebida pela loja deve ser suficiente para atender à sua demanda no próximo round:
   \[
   \sum_{a \in S \cup M} x_{a,m,i,r} \geq O_m(i,(r \mod 17) + 1), \quad \forall m \in M, i \in I_m, r ∈ R
   \]

3. **Capacidade de Transporte**
   - Existe uma capacidade máxima \( C_{a,b} \) para transporte entre dois locais \( a \) e \( b \), limitando a quantidade total de itens por viagem:
   \[
   \sum_{i \in I_a \cap I_b} x_{a,b,i,r} \leq C_{a,b}, \quad \forall a, b \in S \cup M, r ∈ R
   \]

4. **Não Negatividade**
   - As variáveis de decisão devem ser não negativas:
   \[
   x_{a,b,i,r} \geq 0, \quad \forall a, b, i, r.
   \]

## Objetivo do Estudo

O resultado esperado deste estudo é um conjunto otimizado de deslocamentos \( (a, b, I) \) para cada round, indicando quais produtos devem ser transportados e para onde, garantindo o atendimento à demanda e minimizando os custos logísticos. Para isso, serão exploradas abordagens matemáticas e computacionais, considerando:
- Algoritmos exatos e heurísticos para a resolução do modelo.
- Análise de cenários com diferentes padrões de demanda e distribuição de estoque.
- Consideração de restrições adicionais, como prazos de entrega e capacidades dinâmicas de estoque.

Os resultados obtidos poderão contribuir para o aprimoramento de estratégias logísticas em sistemas reais de distribuição de mercadorias, aumentando a eficiência e reduzindo custos operacionais.

