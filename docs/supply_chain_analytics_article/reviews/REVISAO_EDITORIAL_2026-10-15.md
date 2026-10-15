# Revisão editorial — manuscrito PT-BR, 15/10/2026

Escopo: leitura de `manuscript_ptbr.tex` e `references_ptbr.bib`, sem modificar esses arquivos, a dissertação ou resultados. Linhas referem-se à fonte lida no início desta revisão; podem mudar com a escrita paralela. Este é um parecer interno, não revisão da revista. Não executei experimentos nem validei novas estimativas.

**Decisão editorial: revisar antes de apresentar como versão final.** O argumento principal está cauteloso, mas existem erros concretos de espaçamento, notação, definição de siglas e uniformidade de termos. A reescrita deve manter os resultados históricos como históricos. A execução interrompida `overnight_cpu16_2026-10-14` e seus fits preservados não constituem um novo experimento concluído para incorporar às tabelas.

## Correções prioritárias

### E01 — números e símbolos colados ao texto

**Local:** linhas 134–135 e 139.

Trechos: `atraso17`, `até18`, `burstiness$\geq0,30$`, `normalizada$\geq0,05$`, `entropia$\geq2$`.

**Correção:** usar `defasagem de 17 ciclos`, `até 18 ciclos` e espaço explícito entre substantivo e expressão. Preferencialmente definir símbolos e escrever condições inteiras em modo matemático: `$A_{17}\geq0{,}25$ ou $B\geq0{,}30$`; `$|b|\geq0{,}05$ ou $H\geq2$`. Isso evita frases comprimidas e facilita relacionar tabela e fórmulas. Não separar nomes legítimos como M5 ou CV².

### E02 — vírgula decimal em matemática LaTeX

**Local:** linhas 120, 133–137, 161, 182, 186, 201, 213 e 217, incluindo intervalos.

Em matemática, a vírgula sem proteção é pontuação e introduz espaçamento de lista: `$0,70$`, `$1,32$`, `$[-1,442;0,153]$`. Isso torna decimais visualmente ambíguos.

**Correção:** `$0{,}70$`, `$1{,}32$`, `$[-1{,}442;\,0{,}153]$` etc. Alternativa: uma configuração decimal uniforme já compatível com a classe. Não aplicar substituição cega às vírgulas que separam argumentos de `\max`, índices ou itens; elas são separadores reais.

### E03 — resumo e regra usam fronteiras diferentes

**Local:** linha 16 versus equação da linha 114.

O resumo diz atendimento médio **"acima do limiar"**, mas a regra implementada no texto usa $\geq\alpha$.

**Correção:** "entre alternativas cujo fill rate médio é igual ou superior ao limiar". Nas linhas 154–156, distinguir descrição empírica de políticas que estão estritamente acima de 0,70 da definição formal, que inclui igualdade.

### E04 — tempo verbal pode sugerir nova execução dos controladores

**Local:** resumo, linha 16; descrição do portfólio, linha 87; limitações, linha 246.

"Doze políticas são avaliadas em cada base" pode ser entendido como execução atual de todas as políticas. A limitação informa corretamente que saídas históricas estão sendo reanalisadas.

**Correção:** "Reanalisamos matrizes históricas com doze rótulos de política em cada base". Na descrição do método, "as matrizes registram avaliações separadas de cada série". Não acrescentar uma nova avaliação integral concluída, nem mesclar a execução overnight interrompida com os números históricos.

### E05 — siglas sem expansão na primeira ocorrência

**Local:** linha 28 (AIPE e PSE); linhas 39 e 41 (LSTM, GA--RL); linha 87 (EOQ, GA, SA, DQN e PPO); linha 139 (OLS).

**Correção:** expandir AIPE e PSE ao introduzir a proposta; incluir um quadro compacto com identificador, nome e família das doze políticas. Exemplo: "recozimento simulado (SA)" e "mínimos quadrados ordinários (OLS)". Definir o que RL significa na expressão GA--RL. Siglas secundárias, como LSTM, podem receber uma explicação curta na única menção, sem uma subseção teórica que pese o artigo.

### E06 — nome de política muda entre método, tabela e resultados

**Local:** linha 87; tabela nas linhas 165–176; narrativa na linha 201.

O portfólio é introduzido por `BigDataNewsvendor`, `CappedBaseStock` e `Newsvendor`, mas a tabela mostra "Jornaleiro ampliado", "Estoque-base limitado" e "Jornaleiro". O leitor pode entender que se trata de candidatos diferentes.

**Correção:** adotar um único identificador na tabela principal ou apresentar explicitamente a equivalência uma vez: "Jornaleiro ampliado (`BigDataNewsvendor`)". Preservar os identificadores do código em um quadro de rastreabilidade e usar nomenclatura uniforme na redação. Padronizar também GA-DQN/GA--DQN segundo a apresentação tipográfica escolhida.

### E07 — símbolos e domínios ainda não definidos

**Local:** linhas 114, 127 e 139.

As fórmulas de perfil usam $n$, $n_+$, $\mu$, $s$, $\Delta$ e $p_j$ sem definições completas. "Inclui ciclos zero" esclarece $s$, mas não define todos os termos.

**Correção:** acrescentar duas frases: "$n$ é o número de ciclos do cenário, $n_+$ a quantidade de ciclos com demanda positiva e $\mu$ a média da demanda em todos os ciclos; $s$ é seu desvio-padrão amostral." Na expressão de burstiness, definir as distâncias $\Delta$ entre ocorrências e suas média/desvio. Para entropia, definir $p_j$ como frequência relativa no bin $j$. Na seleção, usar `\pi\in\Pi` no domínio do argmin e definir $\Pi$ como o portfólio. Não alterar fórmulas para adequá-las à descrição sem conferir a implementação com o revisor de método.

### E08 — "serviço" alterna entre quatro expressões sem convenção única

**Local:** linhas 16, 103, 112, 164, 182 e 189.

Alternam-se "atendimento", "nível de serviço", "serviço" e "fill rate". O manuscrito faz a distinção matemática correta, mas a variação verbal aumenta o esforço de leitura.

**Correção:** na primeira ocorrência definir "fração de unidades atendidas (fill rate)". Usar "fill rate médio por série" nos cabeçalhos e na comparação principal; reservar "atendimento ponderado pela demanda" para essa agregação específica. Evitar "por volume" sem complemento, que pode parecer uma ponderação da rede.

### E09 — identidade do recorte M5 precisa de uma frase inequívoca

**Local:** linhas 41, 58 e 62.

O texto já evita afirmar reprodução exata, mas "não reproduzem integralmente critérios não especificados" é uma frase indireta. O limiar CV é local e está calculado por produto sobre lojas e ciclos; não se pode insinuar que a coincidência de tamanho certifica os mesmos IDs do artigo-base.

**Correção proposta:** "Este recorte é uma adaptação FOODS com filtro local CV $\geq1{,}5$, inspirada em Zabraoui et al.; a seção 3.7 daquele estudo não publica os IDs nem esse limiar, de modo que a identidade das amostras não foi demonstrada." Conservar 145 IDs históricos BA e 2.408 IDs históricos M5 como universo da reanálise, sujeito à conferência de identidade pelo coordenador. Não substituir por novas coortes sem declarar a mudança.

## Fluidez e precisão dos resultados

- **Linha 16:** "A escolha global muda de VendorResponsive para SA" parece mudança no tempo. Usar "VendorResponsive é a escolha global na base brasileira; SA, no M5".
- **Linhas 182 e 201:** a redação alterna custo com duas casas, variação com três e serviço com quatro casas. Definir convenção de apresentação (por exemplo, variação de custo e pontos percentuais com duas casas; fill rate com quatro). Não recalcular usando as médias já arredondadas: os valores de exibição devem vir dos resultados aprovados.
- **Linha 203:** "redução de custo diferente de zero no sentido dos intervalos" é pouco natural. Usar "Os intervalos condicionais das duas comparações principais incluem zero; os dados não sustentam uma redução de custo positiva nesse critério". Preservar a frase que isso não prova equivalência.
- **Linhas 217 e 255:** o ganho de 17,246% em 0,95 não deve desaparecer por o resultado principal ser negativo. O texto já o relata com queda de serviço e fallback; manter essa separação ao condensar a discussão.
- **Linha 139 versus 141:** o uso do cenário completo é informado duas vezes consecutivas. Manter a definição na subseção e uma explicação breve sobre os rótulos históricos, sem duplicar a advertência.
- **Linhas 39, 45, 146 e 248:** concentrar descrições de implementação em método/limitações. Na revisão de literatura, remover comentários internos sobre grafo de execução que interrompem a comparação entre trabalhos; remeter à seção metodológica.
- **Linhas 60 e 237:** preferir "prefixo sem vendas registradas" a "partida sem histórico" quando não é demonstrado que a ausência representa lançamento real, disponibilidade ou uma situação operacional de cold start. A observação é válida; a interpretação do processo precisa permanecer condicional.

## Referências: verificação editorial e achados

Teste estático das chaves: **10 citações distintas, 10 entradas bibliográficas, nenhuma chave ausente, duplicada ou não utilizada.** Esse teste confirma o encadeamento do arquivo, não a correção factual de todos os metadados.

- **`references_ptbr.bib`, linha 100:** proteger `{M5}` no título para evitar que o estilo numérico reduza a letra a minúscula. Do mesmo modo, preservar siglas e nomes próprios quando um título precisar delas. Não modificar o título bibliográfico para melhorar estilo de redação.
- **Linhas 14, 28, 38, 48, 79, 90 e 101:** um periódico aparece abreviado (`J. Oper. Res. Soc.`) e os demais por extenso. Escolher apresentação uniforme conforme o estilo editorial, sem alterar dados de identificação.
- **Linha 37:** o sobrenome composto está codificado como `Kammouri Alami, Salaheddine`, adequadamente protegido pela forma "sobrenome, nome". Preservar essa forma; não transformá-lo em inicial isolada na reescrita.
- As entradas trazem DOI e número de artigo/páginas. Não identifiquei, nesta revisão editorial, evidência que justifique substituir autores ou DOIs. Uma nova conferência factual de fonte deve ser registrada separadamente, sem alegar que a checagem sintática a realizou.

## Encerramento e QA

O escritor é o responsável exclusivo pelas alterações da fonte. Depois das correções: recompilar BibTeX/LaTeX, conferir ausência de citações indefinidas, renderizar a tabela de perfis e examinar decimais, intervalos, nomes de políticas e larguras de tabela. Não mudar fonte ou margem para acomodar texto. Esta revisão não certifica uma nova versão visual nem substitui os pareceres de método e contribuição. Declarações humanas e direitos de divulgação continuam pendentes antes da submissão.
