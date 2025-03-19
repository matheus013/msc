# Problema de Pesquisa: Otimização do Abastecimento em uma Rede de Lojas com Múltiplos Centros de Estoque e Fábricas

## Descrição Geral

Em uma rede de lojas distribuídas geograficamente, o abastecimento eficiente de mercadorias é um fator crítico para minimizar custos operacionais e garantir a disponibilidade de produtos para os clientes. Essa rede conta com múltiplos centros de estoque e fábricas, cada um armazenando ou produzindo um conjunto específico de itens em quantidades limitadas. Devido às restrições logísticas e ao alto custo de manutenção de grandes estoques nas lojas, essas operam com um inventário reduzido, suficiente para atender à demanda de curto prazo.

Para garantir a operação eficiente das lojas, é necessário planejar quais produtos devem ser transferidos dos centros de estoque para as lojas, garantindo o atendimento à demanda. Além disso, é possível realizar transferências entre centros de estoque e fábricas, caso necessário, para otimizar a distribuição dos itens.

O sistema de abastecimento é baseado em transferências pontuais de mercadorias, sendo que cada transporte tem um custo proporcional à distância percorrida e ao tempo necessário para realização. Cada rota tem um limite máximo de distância que pode ser percorrido por dia. Além disso, as fábricas têm um custo de produção associado a cada tipo de item, e a produção pode ser ajustada conforme a necessidade de reabastecimento dos centros de estoque.

O desafio central é determinar a melhor estratégia de reabastecimento e produção, de forma que o custo total da rede e o tempo total de transporte sejam minimizados enquanto a demanda das lojas é atendida de maneira confiável, respeitando as restrições de tempo e capacidade logística.

## Formulação do Problema

### Parâmetros e Conjuntos
- **\( S \)**: Conjunto de centros de estoque.
- **\( F \)**: Conjunto de fábricas.
- **\( M \)**: Conjunto de lojas.
- **\( I_s \)**: Conjunto de itens disponíveis em um centro de estoque \( s \in S \).
- **\( I_f \)**: Conjunto de itens produzidos por uma fábrica \( f \in F \).
- **\( I_m \)**: Conjunto de itens disponíveis em uma loja \( m \in M \).
- **\( O_m \)**: Demanda esperada para cada loja \( m \in M \).
- **\( E_m \)**: Estoque associado a cada loja \( m \), onde cada loja se relaciona com exatamente um centro de estoque.
- **\( c \)**: Custo unitário de transporte por unidade de distância.
- **\( C_{a,b}\)**: Capacidade máxima de transporte entre dois locais \(a\) e \(b\).
- **\( T_{a,b} \)**: Distância máxima que pode ser percorrida por dia para uma rota entre \( a \) e \( b \).
- **\( p_f(i) \)**: Custo de produção do item \( i \) na fábrica \( f \).
- **\( Q_f \)**: Capacidade máxima de produção da fábrica \( f \).
- **\( t(a, b) \)**: Tempo necessário para transportar produtos entre os locais \( a \) e \( b \).
- **\( d(a, b) \)**: Distância entre os locais \( a \) e \( b \), podendo ser uma fábrica, um centro de estoque ou uma loja.

### Variáveis de Decisão
- **\( x_{a,b,i} \)**: Quantidade do item \( i \) transferida do local \( a \) para o local \( b \).
- **\( y_{f,i} \)**: Quantidade do item \( i \) produzido pela fábrica \( f \).

### Função Objetivo
Minimizar o custo total de transporte e produção e o tempo total de transporte:
\[
\min \sum_{a \in S \cup M \cup F} \sum_{b \in S \cup M} \sum_{i \in I_a \cap I_b} \left( c \cdot d(a, b) \cdot x_{a,b,i} \right) + \sum_{f \in F} \sum_{i \in I_f} p_f(i) \cdot y_{f,i} + \sum_{a \in S \cup M \cup F} \sum_{b \in S \cup M} \sum_{i \in I_a \cap I_b} t(a, b) \cdot x_{a,b,i}
\]

### Restrições

1. **Disponibilidade nos Centros de Estoque e Fábricas**
   - A quantidade de itens transferidos de um centro de estoque não pode exceder sua disponibilidade:
   \[
   \sum_{b \in S \cup M} x_{s,b,i} \leq ê_s(i), \quad \forall s \in S, i \in I_s
   \]
   - A quantidade de itens produzidos não pode exceder a capacidade da fábrica: <!-- capacidade de fabricação infinito -->
   \[
   y_{f,i} \leq Q_f, \quad \forall f \in F, i \in I_f
   \]

2. **Atendimento da Demanda das Lojas** <!-- capacidade de armazenamento infinito -->
   - A quantidade recebida pela loja deve ser suficiente para atender à sua demanda:
   \[
   \sum_{a \in S \cup F} x_{a,m,i} \geq O_m(i), \quad \forall m \in M, i \in I_m
   \]

3. **Capacidade de Transporte** <!-- temos um transporte ideal de via unica, assumindo uma transportadora terceirizada -->
   - Existe uma capacidade máxima \( C_{a,b} \) para transporte entre dois locais \( a \) e \( b \):
   \[
   \sum_{i \in I_a \cap I_b} x_{a,b,i} \leq C_{a,b}, \quad \forall a, b \in S \cup M \cup F
   \]

4. **Restrição de Tempo nas Rotas**
   - A distância percorrida em um dia não pode exceder \( T_{a,b} \):
   \[
   d(a, b) \leq T_{a,b}, \quad \forall a, b \in S \cup M \cup F
   \]

5. **Relacionamento entre Lojas e Centros de Estoque**
   - Cada loja só pode receber produtos do centro de estoque ao qual está associada:
   \[
   x_{s,m,i} = 0, \quad \forall s \in S, m \in M, i \notin I_s, s \neq E_m
   \]
   <!-- - Cada loja deve estar associada ao centro de estoque mais próximo: 
   \[
   E_m = \arg\min_{s \in S} d(m, s), \quad \forall m \in M
   \] -->

6. **Não Negatividade**
   - As variáveis de decisão devem ser não negativas:
   \[
   x_{a,b,i} \geq 0, y_{f,i} \geq 0, \quad \forall a, b, f, i
   \]

