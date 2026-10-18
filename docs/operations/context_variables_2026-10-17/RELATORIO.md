# Contexto demográfico, comercial e socioeconômico disponível

17/10/2026. Inspeção local de fontes e código, sem ajustar modelos, simular políticas ou alterar execuções. Este levantamento mede disponibilidade e variação; não mede ganho preditivo ou operacional.

## O que existe

Os esquemas das fontes brasileiras `seller`, `vendas`, `segmentacao` e `produtos_ativos` oferecem atributos demográficos, comerciais, organizacionais e de produto. Não foram identificados renda domiciliar/pessoal, escolaridade, PIB, IDH ou população municipal nessas fontes. Segmento comercial não deve ser interpretado como renda ou classe socioeconômica.

As 145 séries da coorte Bahia correspondem a 115 revendedores e nove produtos. A junção com o cadastro `seller` encontra exatamente os 115 revendedores, sem multiplicar registros nessa coorte. As 5.741 linhas de venda da coorte contêm os nove atributos abaixo, sem nulos/vazios detectados. Presença no arquivo não comprova disponibilidade histórica.

| Campo | Variação na coorte | Implicação |
|---|---|---|
| `idade` | 23 a 70; 44 valores distintos | Candidata exploratória; falta data de referência da idade. |
| `genero` | 111 F, quatro M | Desequilíbrio limita a avaliação de ganho por categoria. |
| `segmento` | Platina 60, Ouro 32, Rubi 11, Diamante GB seis, Esmeralda GB três, Prata três | Potencial contexto comercial; origem baseada em faixas e regras exige validade temporal e semântica. |
| `filial` | Três valores | Contexto organizacional; cobertura geográfica pequena para generalização entre filiais. |
| `estrutura` | 37 códigos | Semântica precisa ser confirmada; codificação nominal, sem supor ordem ou escala econômica. |
| `ciclos_inativos` | A0 105, A1 oito, A2 um, A3 um | Pouco suporte nas categorias; significado e vigência precisam de confirmação. |
| `status` | Ativo para todos | Sem informação discriminativa nesta coorte. |
| `praca` | Um valor | Sem informação discriminativa nesta coorte. |
| `gerente_regional` | Um valor | Sem informação discriminativa nesta coorte. |

Os números de categorias contam revendedores distintos, não séries nem observações independentes por origem. A idade mediana por linha de venda é ponderada pela frequência de registros e não deve ser apresentada como mediana dos revendedores.

## Disponibilidade histórica não resolvida

Cada atributo possui um único valor por revendedor ao longo das vendas de 202301 a 202504. O cadastro não contém data de vigência ou histórico das mudanças. Há valores já anexados às linhas de vendas dos prefixos das duas origens, mas isso não prova que eram conhecidos na época: pode refletir enriquecimento retrospectivo com um cadastro posterior.

O risco é particularmente importante em segmento, status e atividade, que podem refletir vendas futuras em relação à escolha. A tabela `segmentacao` apresenta faixas comerciais, sem definir sozinha sua janela de apuração histórica ou renda pessoal. Antes de usar esses campos na comparação temporal, é necessário comprovar a referência temporal ou trabalhar com indicadores próprios calculados somente sobre o passado, sem chamá-los de segmento oficial reconstruído. `idade` também requer referência temporal, sem retroceder idades arbitrariamente.

## Atualização da proveniência em 17/10/2026

Depois deste levantamento, o autor confirmou sobre idade, segmento e filial: “sim são do mesmo periodo, eu coletei direto do banco”. A rodada adicional da Bahia utiliza essa declaração de correspondência com o período das vendas como hipótese explícita de admissibilidade. A fonte continua sem histórico independente de alterações por campo. A idade é usada como registrada; não são inventados aniversários, datas de vigência ou valores retroativos. O estudo registra essa limitação e não interpreta segmento comercial como renda pessoal.

Essa confirmação permite preparar a avaliação de desenvolvimento, sem transformar a declaração em uma reconstrução histórica auditada. As fontes e o protocolo próprios da rodada ficam em `experiment/data/context_extension/BA/`.

## Consumo pelo AIPE

O preparador genérico preserva atributos cadastrais e agrega alguns por moda. Isso não equivale a uso pelo experimento canônico. No caminho canônico, `run_aipe_pipeline.py` constrói descritores a partir do prefixo de demanda; `research/pse.py::feature_transform` usa os 17 campos numéricos `FEATURE_COLS` e, opcionalmente, `operational_profile`. As duas implementações preditivas compartilham esse transformador. Idade, gênero, segmento comercial, filial e os demais atributos acima não entram nele.

A agregação genérica por moda sobre uma série completa não deve ser reutilizada como atributo histórico sem novo corte temporal.

## Contexto disponível no M5

`data/m5_walmart/calendar.csv` contém calendário, eventos e `snap_CA`, `snap_TX`, `snap_WI`, sem valores ausentes nesses três indicadores binários. O [guia original da competição](https://storage.googleapis.com/kaggle-forum-message-attachments/772349/15032/M5-Competitors-Guide-Final-10-March-2020.pdf) os define como indicadores de autorização de compras pelo Supplemental Nutrition Assistance Program, programa de assistência alimentar. Não são renda individual dos clientes. Existem também identificadores de estado/loja, atributos de produto e preços semanais na base.

O carregador agregado atual não incorpora SNAP ou eventos como covariáveis do seletor. No M5, `segmento` representa categoria de produto, enquanto na base brasileira representa faixa comercial do revendedor; não são atributos equivalentes. Indicadores de calendário precisam ser vinculados ao estado e às datas reais dos blocos de 21 dias. Valores de eventos futuros só são admissíveis quando conhecidos no instante da decisão; vendas ou preços realizados futuros não se tornam admissíveis por estarem no mesmo arquivo.

## Experimento que responderia à pergunta

Comparar, por base e com mesmos casos, candidatas, regras de seleção e orçamento: contexto de demanda atual; contexto de demanda mais cadastro admissível; e controles por blocos de atributos para identificar o incremento de cada fonte. Na Bahia, testar um conjunto pequeno com suporte suficiente, sem cruzar todas as categorias em perfis esparsos. Validar fora do revendedor e no tempo, mantendo produtos/origens do mesmo revendedor no mesmo grupo e ajustando imputação/codificação dentro de cada partição. Qualquer ajuste de atributos ou margens usa desenvolvimento, não o teste já observado como confirmação independente.

As medidas principais são custo, atendimento, deteriorações por série, frequência de intervenção e concentração dos benefícios. Importância de variável ou acurácia do rótulo não demonstra benefício de reposição. Como a intervenção proposta modifica apenas entradas do PSE, matrizes compatíveis de políticas podem ser reutilizadas e somente os seletores precisam ser ajustados. Mudar previsão interna de controladores, dinâmica ou custos exigiria reavaliar a compatibilidade e refazer ajustes afetados.

Ganho consistente fortaleceria a avaliação do AIPE como metodologia que admite fontes de contexto diferentes. Ausência de ganho também delimitaria o alcance dessa instanciação. Adicionar dados demográficos ou SNAP, isoladamente, não estabelece novidade científica ou benefício.

## Evidência

- Script: `audit_context.py`, neste diretório; DuckDB em memória, um thread e limite de 768 MiB, sem importar o pipeline científico.
- Saída agregada local: `experiment/data/audits/context_variables_2026-10-17/evidence.json`.
- Código de consumo: `experiment/src/simulation/research/pse.py`, `experiment/src/simulation/pipelines/policy_selection/nodes.py`, `experiment/scripts/analysis/run_aipe_pipeline.py` e carregadores de dados.
- A revisão [Retail forecasting: Research and practice](https://www.sciencedirect.com/science/article/pii/S016920701930192X) discute informação contextual e o risco de muitos preditores para pouco suporte. Trata de previsão; não fornece uma garantia de benefício para a seleção de políticas do AIPE.
