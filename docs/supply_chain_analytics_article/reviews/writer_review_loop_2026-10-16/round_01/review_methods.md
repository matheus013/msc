# Revisor de método e evidência — rodada 1

Versão lida: `round_01/source/manuscript_ptbr.tex`, SHA-256 `250a815d12b444dbc36ebcff15a2e8def56e40f8fb8d9989b42d6c231eefd6e3`, e os três fragmentos de resultados do mesmo snapshot. Parecer independente, sem leitura dos pareceres dos colegas nem resposta do escritor.

**Decisão editorial: revisão menor. Pronto cientificamente para submissão: não.** Não encontrei discrepância numérica central nem alegação de superioridade geral escondida. O texto permite circular como manuscrito de desenvolvimento depois de reparos locais; isso não encerra as pendências experimentais abaixo.

## Forças verificadas

- Distingue família de política e calibração individual; global, perfil, rótulo e consequências têm objetivos identificáveis. A ausência de candidata viável permanece registrada.
- Sementes são promediadas por caso; casos repetidos e trajetórias determinísticas não são apresentados como replicações independentes. Bootstrap por loja mantém escolhas e treinamentos fixos e é corretamente qualificado como condicional.
- Informação perfeita, média entre sementes e realização individual são distinguidas. O limite físico não é tratado como erro do seletor.
- A extensão com recuo é explicitamente desenvolvimento no próprio piloto; concentração de 92% da economia e apenas três intervenções aparecem no resumo, resultados e discussão.
- Conferência documental: `numerical_audit.json` da auditoria numérica Bahia registra 4.350 ajustes, 290 casos, 283 serviços definidos, aumento de custo 1,9259705% e serviço +3,8869703 p.p., compatíveis com o arredondamento do artigo. `reviews/pse_consequences_2026-10-15/summary.csv` registra global 1.086,97465/0,714572 e consequências com recuo 1.070,03401/0,717248, 47 viáveis, três mudanças e nenhuma nova falha ou recuperação, compatíveis com o fragmento. Não executei treinamento, retreinamento ou modificação de resultados.

## Reparos editoriais

### MTH-01 — menor — objetivo de calibração insuficientemente explícito

**Local:** manuscrito, seção “Simulador, portfólio e calibração”, linhas 133–135.

**Evidência:** “A calibração pode penalizar déficit de serviço” não identifica quais métodos efetivamente o fazem nem distingue penalidade de restrição rígida. O snapshot histórico Bahia em `code_snapshot/experiment/src/simulation/core/policies.py`, `_eval_static`, usa custo acrescido de `30 max(0;0,70−F) max(C;1)`; `rl_torch.py`, `_terminal_deficit_penalty`, e `research/hybrid.py`, `episode`, também aplicam penalidade terminal. A calibração vetorizada da Capped usa escala intrínseca à série e termos de risco. Portanto, seria incorreto simplificar RL como recompensa exclusivamente de custo, mas também é impreciso deixar toda a diferença como uma possibilidade indeterminada.

**Reparo verificável:** substituir a frase vaga por uma descrição compacta da penalização efetiva de GA, RL/híbrido e Capped, com peso 30 e a escala usada, consultando os snapshots associados a cada matriz. Explicitar que a penalidade aproxima o objetivo na calibração e não garante F≥0,70; a seleção posterior aplica sua regra de viabilidade à validação. Não é necessário inserir um catálogo de hiperparâmetros no corpo.

**Verificação:** cada afirmação corresponde à função/configuração congelada e o leitor não confunde custo reportado, aptidão de treino e regra do seletor. Classificação: `writing`; aberto.

### MTH-02 — menor — frase pode sugerir ajuste sobre o teste

**Local:** manuscrito, “Substituição dos métodos e decisão de intervenção”, linha 82.

**Evidência:** “Todas as candidatas são ajustadas nas unidades de teste” usa “teste” para a identidade da série, mas o leitor costuma entender bloco temporal. O restante do texto afirma corretamente ajuste no prefixo histórico.

**Reparo verificável:** escrever “Nas séries avaliadas, todas as candidatas são calibradas exclusivamente no prefixo de treino; suas trajetórias no teste permitem medir as recomendações e construir diagnósticos retrospectivos”, ou equivalente.

**Verificação:** informação futura não parece entrar no ajuste, e a despesa experimental continua distinguida do uso pretendido. Classificação: `writing`; aberto.

### MTH-03 — menor — fixar o alcance da palavra global e da média

**Local:** manuscrito, “Custo, serviço e escolha”, linhas 95–97, e “Avaliação e desenvolvimento do piloto”, linha 138.

**Evidência:** a global é escolhida na validação das outras lojas por origem/fold; pode variar entre lojas reservadas e origens. As barras sobre custo e serviço são usadas antes de explicitar a média aritmética dos casos, após média entre sementes. O texto distingue atendimento ponderado depois, mas uma frase junto à regra elimina a ambiguidade.

**Reparo verificável:** dizer que “global” significa família comum às séries recomendadas naquele fold/origem, não uma única vencedora escolhida para todos os folds. Definir média como aritmética por caso série–origem, com sementes primeiro agregadas e serviços indefinidos excluídos; esclarecer que isso não garante serviço individual.

**Verificação:** conceito permanece compatível com `best_policy`, exclusão por loja e resultados; nenhuma contagem ou número muda. Classificação: `writing`; aberto.

## Pendências científicas que não se resolvem por redação

### MTH-04 — maior — confirmação independente do recuo

**Local:** desenho experimental, extensão por consequências, conclusão.

O piloto já orientou a criação das variantes; três decisões, concentradas em duas lojas, sustentam o resultado favorável. A validação aninhada protege a escolha da margem dentro da implementação, mas não desfaz a reutilização do piloto para decidir a própria implementação. O artigo já reconhece corretamente essa condição. Para submissão, fixar a implementação e o protocolo antes de avaliar unidades/períodos reservados ao desenvolvimento, registrar influência do piloto e reportar deteriorações e concentração junto às médias. Não iniciar nada neste ciclo. Classificação: `experiment`; aberto, redação já adequadamente limitada.

### MTH-05 — maior — robustez e calibração comparável

**Local:** calibração e limitações.

O próprio artigo registra penalização adicional da Capped na Bahia e ausência desses termos no M5, orçamento desigual e ausência de sensibilidades de custo/prazo/horizonte. Isso limita interpretar a metodologia em aplicações além da instanciação. São necessárias avaliação harmonizada e sensibilidades justificadas antes de alegações amplas; não exigir que este ciclo editorial invente ou antecipe resultados. Classificação: `experiment`; aberto, limites corretamente reconhecidos.

## Mapa de alegações

| Alegação | Situação |
|---|---|
| Implementações do AIPE podem ser comparadas na matriz comum | Sustentada para interfaces exercitadas |
| Perfil na Bahia acrescenta atendimento com custo pontual maior | Sustentada condicionalmente à matriz e população descritas |
| XGBoost por classe perde no piloto | Sustentada neste piloto |
| Consequências com recuo melhoram as médias observadas | Sustentada como resultado de desenvolvimento, concentrado |
| Recuo melhora fora do desenvolvimento ou oferece garantia | Não demonstrada; o artigo não a afirma |
| Portabilidade integral, economia computacional e robustez de aplicação | Necessitam avaliação adicional |

Declarações autorais, tradução e pacote de reprodução continuam pendências de submissão já explícitas. Não recomendo ampliar o texto para repetir as limitações: os três reparos são locais e podem substituir frases existentes.
