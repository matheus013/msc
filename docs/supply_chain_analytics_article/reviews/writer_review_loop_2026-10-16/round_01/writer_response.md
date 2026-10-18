# Rodada 1 — revisão do escritor

Status: pronto_para_revisao. Revisão substantiva anterior aos três pareceres; sem compilação nesta tarefa.

## Esboço argumentativo

1. **Problema:** escolher famílias de reposição sob demandas heterogêneas, distinguindo escolha de família e calibração por série.
2. **Dificuldade:** transformar evidência retrospectiva em recomendação anterior ao teste e identificar a origem de cada falha.
3. **Proposta:** cinco etapas do AIPE ligam contexto, evidência, PSE, decisão de intervenção e diagnóstico. A regra de intervenção é uma decisão avaliável separadamente da candidata proposta.
4. **Posicionamento:** classificação e inventário, seleção de algoritmos, consequências e recuo são antecedentes. O trabalho formula uma articulação para reposição e avalia implementações internas; não demonstrou prioridade nem superioridade frente às integrações próximas.
5. **Evidência:** Bahia responde sobre perfil; M5 compara perfil e preditivos, examina regra de intervenção e decompõe falhas. Matrizes congeladas mantêm resultados positivos e negativos.
6. **Interpretação:** identificar quanto potencial retrospectivo se converte em benefício e onde revisar a metodologia. Ganho concentrado no piloto é desenvolvimento exploratório.
7. **Alcance:** limites empresariais, estatísticos e de validação; ampliação ainda fora dos resultados, seguida de conclusão proporcional à evidência.

## Alterações efetivas

- Introdução reescrita para apresentar o mecanismo de decisão antes das implementações, com encadeamento representação → evidência → intervenção → diagnóstico.
- A contribuição passa a ser identificada pelo que o procedimento torna verificável; compatibilidade de interfaces e ganho operacional são critérios distintos.
- Trabalhos relacionados agora conectam cada antecedente ao papel na metodologia, reduzindo frases defensivas isoladas. A ausência de comparação empírica com integrações próximas continua explícita.
- Explicado o indicador de intervenção e a relação entre recomendar uma candidata e autorizar sua aplicação.
- Substituídas expressões de engenharia de software por condições científicas: matriz completa, informação disponível, decisões fixadas e identificação de cada avaliação.
- Perguntas experimentais articuladas às aplicações e às implementações que as respondem.
- Retirada do método a subseção sobre execução ampliada pendente: seus aspectos científicos foram condensados nas limitações/próximos testes. Informações quantitativas e mudanças de desenho relevantes foram conservadas; detalhes de checkpoints retirados.
- Discussão reorganizada para interpretar potencial de seleção, transferência, concentração das intervenções e origem das falhas. A conclusão retoma as cinco etapas e a evidência efetivamente produzida.
- Resultados, figuras, bibliografia, highlights, dissertação e experimentos não foram alterados.

## Esboço reverso das seções alteradas

- **Introdução:** decisão operacional → distinção entre seleção e calibração → dificuldade de usar evidência e diagnosticar perdas → AIPE → contribuição → papel e síntese das aplicações.
- **Relacionados:** classificação/desempenho → decisão na revista-alvo → portfólio híbrido → calibração e avaliação → antecedentes de intervenção/efeitos previstos.
- **Método:** produto da metodologia → cinco etapas → contratos → substituição → candidata/intervenção → custo de avaliação versus aplicação → integridade da comparação.
- **Desenho:** três perguntas → bases → simulador/calibração → transferência/desenvolvimento → oráculos. A ampliação pendente deixa de interromper o desenho que efetivamente gerou os resultados.
- **Discussão:** potencial e benefício → Bahia → dificuldade preditiva → efeito e concentração da intervenção → diagnóstico → suporte/transferência.
- **Limitações/conclusão:** limites observacionais e de simulação → limites de modularidade/confirmação → alcance da ampliação → contribuição e implicações sustentadas.

## Mapa de alegações e evidências

| Alegação | Evidência | Situação |
|---|---|---|
| AIPE especifica um procedimento em cinco etapas com decisão explícita de intervenção | Método, contratos e equação de escolha | Sustentada como formulação |
| Implementações distintas podem ser examinadas sob responsabilidades comuns | Sete candidatas e implementações do PSE nas matrizes | Sustentada no escopo exercitado; portabilidade integral não demonstrada |
| Seleção por perfil acrescenta atendimento com custo pontualmente maior na Bahia | Tabela por origem e fragmento Bahia: +3,89 p.p.; +1,93% | Sustentada na calibração reportada |
| Preditivo por classe perde no piloto M5 | Tabela M5: +13,9% de custo; −4,11 p.p. | Sustentada no piloto |
| Consequências com recuo produz ganho concentrado | Fragmento de consequências: −1,56%; +0,27 p.p.; três intervenções; ~92% em uma série | Sustentada como desenvolvimento exploratório |
| Diagnóstico separa intervenções sobre seleção, portfólio e condições iniciais | Oráculos e decomposição M5 | Sustentada retrospectivamente |
| Originalidade comparativa, melhoria generalizável e economia de implantação | Comparadores próximos não reproduzidos; ausência de confirmação independente/implantação | Não demonstradas; não reivindicadas |

## Autocheck adversarial

- **Contribuição:** proposta identificável pelo mecanismo, sem alegar novidade por nomear módulos. Comparação com antecedentes próximos continua pendência científica.
- **Clareza:** conceitos são definidos antes das variantes; uma regra pode recomendar bem e intervir mal, ou conservar a referência. Nenhum jargão de fila foi introduzido.
- **Força experimental:** preservados deterioração do preditivo, trade-off Bahia, concentração da economia e caráter exploratório. Nenhuma nova conclusão quantitativa.
- **Completude:** ressalvas sobre unidades, sementes, restrição média/individual, informação futura, calibrações distintas e ausência de confirmação mantidas.
- **Solidez:** integridade informacional e desenho de transferência conservados. Custo computacional de implantação não apresentado como demonstrado.
- **Equações/citações/números:** comparação automática com round_00 encontrou equações destacadas e grupos de citações idênticos; nenhum token numérico foi adicionado ou removido. Os três fragmentos de resultados não foram editados.
- **Verificação restante:** coordenador compila e congela; três revisores julgam a mesma versão e registram achados. Esta autocheck não substitui sua revisão.
