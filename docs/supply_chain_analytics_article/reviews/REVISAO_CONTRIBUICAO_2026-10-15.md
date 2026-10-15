# Revisão de contribuição e sustentação — 15/10/2026

Parecer interno, somente leitura do manuscrito `manuscript_ptbr.tex`. Linhas correspondem à fonte consultada durante a escrita paralela e podem mudar. Não executei experimentos nem alterei manuscrito, código ou dissertação. A execução interrompida e seus 19.818 ajustes preservados não são uma nova rodada completa.

**Decisão: contribuição defensável como reanálise exploratória, ainda insuficiente para apresentar o protocolo corrigido como validado.** O texto atual já informa a separação Bahia/M5, as limitações do M5, a ausência de validação do PSE e a falta de reprodução exata de Zabraoui. Não encontrei justificativa para acrescentar reivindicação de novidade algorítmica ou superioridade geral de GA–DQN.

## Ajustes possíveis agora, sem novos experimentos

| Item | Local atual | Achado e correção proposta |
|---|---|---|
| C01 — identidade da evidência | 16, 50, 87–94 | “Doze políticas são avaliadas” pode sugerir nova execução controlada. Preferir “reanalisamos matrizes históricas com doze rótulos de política”. A rastreabilidade da reanálise não certifica os controladores originais; o texto admite ausência de manifesto histórico completo. |
| C02 — contribuição central | 28–34, 47, 258 | Apresentar como análise empírica do ganho incremental de perfis contra uma referência global viável, condicionada a duas matrizes. A distinção conceitual entre heterogeneidade e utilidade do seletor é o argumento do artigo, não um novo teorema ou método de seleção. “Testar em outro conjunto” deve significar repetir a reanálise, sem sugerir teste prospectivo independente. |
| C03 — serviço e agregação | 16, 103, 110–119 | A fração de unidades atendidas é calculada dentro da série, mas a seleção usa média não ponderada entre séries. “Por volume” no resumo pode levar a interpretar uma restrição agregada da rede. Usar “fill rate médio por série” na definição operacional principal. |
| C04 — prefixo sem vendas | 61, 156, 238 | As contagens de zeros estão demonstradas; não está demonstrado que identifiquem lançamento, indisponibilidade ou verdadeiro início operacional. Preferir “prefixo sem vendas registradas” a “cenário de partida sem histórico”. Sua relação com desempenho de SA/GA–DQN permanece hipótese, não explicação causal confirmada. |
| C05 — contexto do projeto | 28, 32, 39, 249, 253 | AIPE/PSE, piloto da Paraíba e menções ao grafo LSTM podem ser condensados. Não são contribuições validadas pelos resultados apresentados. Uma frase sobre a origem na qualificação e outra delimitando a seleção por regras bastam; a exclusão de módulos pertence ao método/limitações. |
| C06 — resultado negativo com cuidado | 203, 234, 256 | Manter “não estabelece vantagem geral de custo no limiar principal”. Não substituir por “perfis não funcionam” ou “políticas são equivalentes”: os intervalos não demonstram equivalência, e alguns limiares apresentam contrastes condicionais não nulos. |
| C07 — comparação entre bases | 32, 47, 63, 238 | Bahia e M5 são estudos separados, não observações intercambiáveis de uma mesma experiência. O segundo conjunto amplia o contraste de origem e estrutura, mas diferenças de orçamento, horizonte e preparação impedem atribuir a troca de vencedor somente ao país, perfil ou característica de demanda. |
| C08 — viabilidade e sensibilidade | 119, 218 | Conservar a distinção entre atendimento médio no conjunto de escolha, atendimento da loja excluída e garantia individual. Em 0,95 há fallback por falta de alternativa admissível; esse contraste não pode aparecer em resumo/conclusão como economia mantendo a mesma restrição em todos os grupos. |

## Separação obrigatória entre histórico e protocolo futuro

- **Histórico apresentado:** Bahia com 145 séries e M5 com 2.408, doze rótulos de política, prefixos de referência 17 e 19. Sem alterar números ou tabelas a partir de resultados parciais posteriores.
- **Validação em preparação:** seis políticas principais e controle DQN para ablação do híbrido; Bahia em origens 17/23 e M5 em 50/64/78. As elegibilidades M5 informadas pelo coordenador são 885/2.033/2.408 séries por origem. São contagens de casos por janela, não três bases independentes nem uma substituição automática das 2.408 séries históricas.
- **Nacional:** não fornece resultado concluído a este manuscrito. A qualificação e a rodada interrompida também não autorizam declarar validação nacional.
- **Zabraoui:** o vínculo bibliográfico permite discutir a inspiração FOODS e GA–DQN; não certifica IDs, corte CV local de 1,5, transformação ou reprodução numérica. O esclarecimento atual na linha 59 é adequado e deve permanecer.

Não é necessário inserir todos esses detalhes futuros no corpo do artigo agora. Um quadro interno de proveniência pode registrar o plano; somente uma rodada encerrada e auditada justifica substituir a evidência histórica.

## O que exige evidência nova

1. Ganho do portfólio reduzido ou superioridade de GA–DQN: requer protocolo corrigido, componentes GA/DQN, sementes e condições comparáveis; a tabela histórica não valida versões atuais.
2. Utilidade prospectiva dos perfis: requer cálculo de características e filtros somente no passado e seleção fora do teste.
3. Explicação causal para o resultado do M5: requer análise controlada de prefixos nulos, inicialização e orçamento. Contar zeros não basta.
4. Recomendação operacional generalizável: requer critérios de custo e atendimento coerentes, restrições relevantes e limites de dependência entre séries e lojas.

**Prioridade editorial:** corrigir ambiguidade de proveniência e agregação, manter resultados históricos intactos e reduzir contexto interno do projeto. Isso melhora clareza e honestidade da contribuição sem fabricar um resultado novo.
