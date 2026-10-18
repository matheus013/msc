# Parecer independente — método e evidência, rodada 1

16/10/2026. Revisão somente de leitura das fontes científicas, com `academic-paper-reviewer` e a lente de coerência/escopo do `paper-writing/author_profile/gate_semantic.md`. Não executei treino, simulação ou alteração da fila. Este parecer é assistência editorial interna, sem previsão de aceite.

## Veredito

**Aprovado editorialmente no escopo desta revisão. Zero achados críticos ou maiores e nenhum reparo textual obrigatório.** O novo contrato gerencial e temporal está distinguido da implementação e da evidência efetivamente produzida. As lacunas empíricas declaradas permanecem abertas; não há prontidão científica para submissão demonstrada por este parecer.

Todos os 36 arquivos do manifesto conferem por SHA-256, inclusive as fontes, os PDFs e os documentos-base. Conferi o artigo completo em LaTeX, seus três fragmentos quantitativos por referência ao manifesto, os capítulos designados da dissertação, a auditoria das entradas, o protocolo e o plano/planejador sequencial. A extração dos PDFs confirma a presença dos trechos metodológicos principais: artigo com 24 páginas e dissertação com 66. A inspeção visual integral pertence ao revisor editorial/coordenador; extração textual não a substitui.

## Mapa de evidência e guardrails

| Questão | Localização exata | Evidência e julgamento |
|---|---|---|
| Custos de cenário | Artigo, linhas 100 e 175; dissertação `instanciacao_aipe.tex:9` e `:163` | h=1, p=5, K=50, c=0,5 não são custos financeiros calibrados. A discussão impede converter ganhos em economia realizada em reais. Compatível com a auditoria e `protocol.py:calibration_config`. |
| Prazo de reposição | Artigo `manuscript_ptbr.tex:136`; dissertação `instanciacao_aipe.tex:7` | Dois ciclos, convenção exata; não confunde dois ciclos comerciais com 42 dias Bahia. O plano sequencial preserva períodos nativos. |
| Estado inicial comum | Artigo `:138`; dissertação `instanciacao_aipe.tex:102` | Aquecimento EOQ e trajetória EOQ de validação produzem condições comuns, sem continuidade própria de cada candidata. Confirmado em `protocol.py:common_initial_state`. |
| Perfil derivado versus externo | Artigo `:75`; dissertação `capitulos/proposta.tex:11` e `instanciacao_aipe.tex:45` | Contrato admite perfil externo, mas resultados exercitam os derivados do histórico. `profile_of(train)` e worker corroboram; não há alegação de perfil humano validado. |
| Dados ausentes Bahia | Artigo `:116`; dissertação `instanciacao_aipe.tex:41` | Ausência de agregado preenchida com zero é hipótese declarada; 2.697/5.510 é descrito como auditoria canônica, sem rebatizar os resultados históricos. |
| Estado empresarial observado | Artigo `:192`; dissertação `instanciacao_aipe.tex:163` | Fotografia de maio/2025 é posterior aos testes, com vínculos múltiplos. Não é tratada como saldo inicial histórico. |
| Entrega gerencial parcial | Artigo `:75`; dissertação `instanciacao_aipe.tex:167`, `capitulos/proposta.tex:13` | Saída pretendida contém parametrização, mas persistência atual não reúne parâmetros finais de todas as políticas, redes reutilizáveis, trajetórias e componentes de custo. Não são ordens executáveis. |
| Decisão humana | Artigo `:52`, `:114` e `:192`; dissertação `capitulos/proposta.tex:13` | Apoio e aprovação humana são requisitos pretendidos; eficácia com usuários não é apresentada como demonstrada. |
| Cadência parametrizável | Artigo `:77`, `:91`, `:194`; dissertação `capitulos/proposta.tex:25–29` e `instanciacao_aipe.tex:143` | Período, agregação, revisão, horizonte, janela e atualização são distintos. Família permanece por sete ciclos no desenho avaliado. Reposição a cada período não é nova seleção PSE. |
| Matrizes históricas e harmonizadas | Artigo `:146` e `:196`; dissertação `instanciacao_aipe.tex:130–139` e `capitulos/plano_continuidade.tex:15` | Penalidades históricas da CappedBaseStock são explicitadas; configuração harmonizada é desenvolvimento separado. Tabelas não são anunciadas como oriundas da rodada unificada. |
| Relevância e não inferioridade | Artigo `:175`; dissertação `capitulos/proposta.tex`, seção Critérios de avaliação | Julgamento contextual não substitui evidência; ausência de significância não é equivalência. Sem promessa de redução universal nem tolerância formal inventada. |
| Planejamento sequencial | `PLANO_AVALIACAO_SEQUENCIAL_AIPE.md`, seções O que foi preparado e Por que a execução ainda está bloqueada; `plan_aipe_sequential.py:177–193` | Agenda ilustrativa e estado NOT_READY, sem treino nem nova simulação. Mantém estoques/trânsito por estratégia e explicita maturação de rótulos. O horizonte solicitado pode exceder acompanhamento, mas o período pontuável é limitado e declarado. |

## Questões empíricas abertas, sem novo erro textual

1. Confirmar regra principal de intervenção em reserva genuinamente não utilizada; piloto observado permanece desenvolvimento. Exclusão de loja e separação de IDs não certificam independência confirmatória.
2. Completar calibração harmonizada e analisar sensibilidades de custos/prazo/horizonte antes de ampliar a conclusão.
3. Persistir os artefatos necessários e implementar continuidade/adaptação do estado antes de avaliar revisão cíclica. Agenda não demonstra dinâmica de estoque.
4. Validar a qualidade de zeros e a correspondência vendas–demanda; calibrar custos/estoques operacionais para alegar benefício financeiro de campo.
5. Avaliar compreensão, uso e eventual ajuste das recomendações por gerentes antes de alegar eficácia humana.
6. Delimitar comparações com antecedentes metodológicos próximos e manter análise de concentração/deteriorações e dependência entre casos.

Esses pontos já estão delimitados pelos textos e pelo protocolo. Não os converti artificialmente em achados editoriais maiores; sua resolução exige evidência e decisões de desenho, não outra paráfrase.

## Fechamento e escopo da aprovação

Afirmações sustentadas: formulação metodológica, interfaces experimentadas, seleção por contexto em blocos fixos, métricas simuladas e diagnóstico condicionado ao portfólio/ambiente. Afirmações não demonstradas: economia monetária realizada, eficácia de supervisão humana, compras executáveis, revisão dinâmica validada ou superioridade universal.

Aprovação vinculada aos hashes de `round_01/manifest.json`. Qualquer alteração substantiva de escopo/estado empírico exige nova leitura dos trechos afetados. Não houve alteração de fontes pelo revisor.
