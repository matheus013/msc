# Alterações de redação — manuscrito em português

Data: 15/10/2026. Escopo: `manuscript_ptbr.tex`. Revisão textual; nenhum PDF foi compilado ou modificado neste trabalho.

## Correções realizadas

- Corrigidos `atraso17` e `até18`: a característica passa a ser descrita como autocorrelação na **defasagem 17**, com retorno zero até **18 ciclos**, mantendo a convenção histórica descrita. Incluído espaço antes dos limiares matemáticos na tabela de perfis.
- O resumo agora informa explicitamente que se trata de reanálise de matrizes históricas. A regra de viabilidade é **igual ou superior ao limiar**, coerente com a fórmula, em vez de apenas “acima”.
- Explicitado que Bahia e M5 são experimentos separados, com ambiente por série e seleção dentro de cada base. A seção de resultados foi identificada como histórica; novas execuções não concluídas não fazem parte das conclusões.
- M5 mantém os números históricos de 2.408 séries, dez lojas e 287 produtos. FOODS e CV 1,5 são identificados como adaptação local inspirada em Zabraoui. A seção 3.7 do artigo de referência não fornece IDs ou limiar numérico para certificar amostra idêntica.
- A limitação de extremos por cercas de IQR foi descrita com mais precisão, sem apresentá-la como suavização temporal.
- Os nomes BigDataNewsvendor, CappedBaseStock e Newsvendor substituem traduções alternadas na tabela, alinhando-a ao portfólio declarado.
- A escolha entre CPU e GPU passa a depender de medição do tempo total; retirou-se a expectativa genérica de aceleração por GPU.

## Preservação e verificação

Nenhum custo, fill rate, intervalo, contagem histórica, horizonte ou orçamento de treinamento foi alterado. Permanecem as doze políticas das matrizes antigas: o portfólio candidato de seis políticas não foi apresentado como resultado concluído.

A inspeção estática confirmou que todas as citações e referências cruzadas da fonte encontram suas entradas/labels e que as duas colagens indicadas pelo usuário não permanecem. Essa inspeção não substitui compilação e conferência visual do coordenador.

As revisões anteriores `editorial_ptbr.md`, `methods_ptbr.md` e `contribution_ptbr.md` foram consultadas. Achados adicionais dos pareceres de 15/10/2026 serão incorporados apenas quando sustentados pela evidência disponível.

## Ajustes a partir das revisões de 15/10

- Expandida a primeira apresentação pertinente de AIPE, PSE, GA, DQN, SA, EOQ e PPO; no resumo, SA foi escrito como recozimento simulado.
- Corrigida a vírgula decimal em expressões matemáticas para a forma agrupada do LaTeX (`0{,}70`), sem alterar os valores.
- A definição de fill rate agora exige demanda acumulada positiva. O caso indefinido de janela sem demanda deve ser reportado separadamente; o texto registra que as séries históricas têm demanda positiva no horizonte posterior ao prefixo, sem recalcular ou modificar suas métricas.
- Mantida a distinção entre implementação histórica e código atual. Questões do híbrido detectadas na auditoria do novo protocolo não foram atribuídas automaticamente aos controladores que produziram as matrizes antigas. Não há alegação de reprodução exata de Zabraoui ou de superioridade causal do híbrido.

- Incorporadas as recomendações editoriais E04/E07: doze rótulos descritos como avaliações arquivadas; definidos portfólio, número de ciclos, ciclos positivos, média, desvio, intervalos entre ocorrências e frequências do histograma. Expandido OLS e esclarecida a única menção inicial a LSTM.
- Retirada a interpretação automática de prefixo sem vendas como lançamento/cold start real; o texto mantém a observação empírica e distingue a interpretação operacional ainda não demonstrada.
- A frase sobre intervalos que incluem zero foi simplificada, sem afirmar equivalência entre estratégias. Eliminada a repetição consecutiva sobre perfis calculados no cenário completo.
- Protegida a sigla `{M5}` no título da entrada bibliográfica. Nenhum autor, DOI ou resultado foi modificado.

## Fechamento após os três pareceres novos

Lidos `REVISAO_EDITORIAL_2026-10-15.md`, `REVISAO_METODOS_2026-10-15.md` e `REVISAO_CONTRIBUICAO_2026-10-15.md`. O resumo passou a usar explicitamente **fill rate médio por série**; a comparação M5 é descrita como retrospectiva. Acrescentada a ressalva de que manter IDs filtrados pela trajetória completa permite avaliação temporal condicional à coorte histórica, sem tornar prospectivo todo o processo de formação da amostra. A continuidade prevê a comparação do híbrido com GA e DQN isolados, sob mesmas condições e com esforço computacional total informado.

Foram lidas as três inconsistências do caminho atual GA-DQN (ação das demonstrações, recompensa terminal e diferença entre treinamento e política aplicada). Elas permanecem pendências de código do protocolo futuro, não fatos certificados sobre a geração das matrizes antigas; não se alterou o modelo histórico nem se alegou vantagem causal ou implementação exata de Zabraoui. As diferenças entre painel histórico de 93 ciclos e painel futuro de 92 blocos não foram mescladas às tabelas.

**Fonte estável para compilação pelo coordenador.** Edições encerradas; nenhum PDF compilado por este agente.
