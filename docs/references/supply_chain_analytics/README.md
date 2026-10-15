# Coleção Supply Chain Analytics

Coleta em 14/10/2026. ISSN 2949-8635, Elsevier. Sem filtro temático: o objetivo desta etapa é volume e cobertura.

**257 registros catalogados; 21 artigos únicos reunidos em PDF.** Destes, 19 têm a versão publicada e 2 têm versões de manuscrito. Dois artigos já existiam no projeto; 19 artigos são novos no acervo.

## Arquivos

- `pdfs/`: um PDF principal por DOI, organizado por ano.
- `versions/`: outras versões preservadas, sem contar como novos artigos.
- [catalog.csv](catalog.csv): título, DOI, ano, volume, status, versão, páginas e origem.
- [catalog.json](catalog.json): catálogo completo, hashes e histórico de tentativas.
- [references.bib](references.bib): metadados bibliográficos de todos os registros.
- `metadata/`: respostas originais dos índices e fontes adicionais.

## Cobertura e limitações

A identificação cruzou Crossref (225 registros), OpenAlex (249) e DOAJ (186). As bases se sobrepõem; a união contém 257 DOIs. Registros editoriais e sumários permanecem no catálogo, mas não são contados como artigos baixados.

O acesso HTTP automatizado ao ScienceDirect respondeu 403. A API de texto completo da Elsevier, testada com a chave fornecida, respondeu 403: `Requestor configuration settings insufficient for access to this resource`. Os lotes preparados pelo navegador não produziram arquivos locais verificáveis. A chave não foi salva nesta pasta nem no código.

Os PDFs foram obtidos de repositórios públicos dos autores/instituições, e dois artigos foram incorporados a partir de cópias já existentes no projeto. Cada arquivo foi aberto, conferido contra o título/DOI e registrado com SHA256. Não foram usados PDFs de outras revistas para aumentar a contagem.

Há 215 registros sem fonte alternativa de PDF identificada e 21 com fontes que não disponibilizaram um arquivo nesta sessão (restrição, timeout, link desatualizado ou identidade não confirmada). Isso não significa que os artigos não existam ou sejam fechados; a coleta permanece incompleta.

## Retomar a coleta

Na raiz do repositório, usando o ambiente preparado:

```powershell
.\experiment\.venv\Scripts\python.exe experiment/run.py collect-sca
# Repetir falhas após corrigir acesso ou acrescentar fontes:
.\experiment\.venv\Scripts\python.exe experiment/run.py collect-sca --retry-failed
```

As fontes adicionais ficam em `metadata/extra_sources.json`. O script retoma os PDFs existentes e não apaga os artigos anteriores. Para instalar as dependências de coleta: `pip install -e ".[research]"` dentro de `experiment/`.

Fontes: [revista no ScienceDirect](https://www.sciencedirect.com/journal/supply-chain-analytics), [ISSN Portal](https://portal.issn.org/resource/ISSN/2949-8635), [DOAJ](https://doaj.org/toc/2949-8635), [documentação da API Elsevier](https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl).

## PDFs por ano

| Ano | Artigos únicos com PDF |
|---|---:|
| 2023 | 9 |
| 2024 | 4 |
| 2025 | 4 |
| 2026 | 4 |

## Índice dos arquivos disponíveis

| Ano | Artigo | DOI | Versão | Páginas |
|---|---|---|---|---:|
| 2023 | [A two-stage data envelopment model for evaluating the exchange efficiency of the imports and exports in a railway supply chain network](pdfs/2023/100004_A_two_stage_data_envelopment_model_for_evaluating_the_exchange_efficiency_of_the_impo.pdf) | [10.1016/j.sca.2023.100004](https://doi.org/10.1016/j.sca.2023.100004) | publishedVersion | 13 |
| 2023 | [A novel grey multi-objective binary linear programming model for risk assessment in supply chain management](pdfs/2023/100012_A_novel_grey_multi_objective_binary_linear_programming_model_for_risk_assessment_in_s.pdf) | [10.1016/j.sca.2023.100012](https://doi.org/10.1016/j.sca.2023.100012) | publishedVersion | 25 |
| 2023 | [A supply chain risk assessment index for small and medium enterprises in post COVID-19 era](pdfs/2023/100023_A_supply_chain_risk_assessment_index_for_small_and_medium_enterprises_in_post_COVID_1.pdf) | [10.1016/j.sca.2023.100023](https://doi.org/10.1016/j.sca.2023.100023) | publishedVersion | 12 |
| 2023 | [A new key performance indicator model for demand forecasting in inventory management considering supply chain reliability and seasonality](pdfs/2023/100026_A_new_key_performance_indicator_model_for_demand_forecasting_in_inventory_management_.pdf) | [10.1016/j.sca.2023.100026](https://doi.org/10.1016/j.sca.2023.100026) | publishedVersion | 15 |
| 2023 | [A multi-agent-based real-time truck scheduling model for cross-docking problems with single inbound and outbound doors](pdfs/2023/100028_A_multi_agent_based_real_time_truck_scheduling_model_for_cross_docking_problems_with_.pdf) | [10.1016/j.sca.2023.100028](https://doi.org/10.1016/j.sca.2023.100028) | manuscript_version_unconfirmed | 25 |
| 2023 | [A fuzzy TOPSIS model for selecting digital technologies in circular supply chains](pdfs/2023/100038_A_fuzzy_TOPSIS_model_for_selecting_digital_technologies_in_circular_supply_chains.pdf) | [10.1016/j.sca.2023.100038](https://doi.org/10.1016/j.sca.2023.100038) | publishedVersion | 16 |
| 2023 | [A hybrid forecasting model with logistic regression and neural networks for improving key performance indicators in supply chains](pdfs/2023/100041_A_hybrid_forecasting_model_with_logistic_regression_and_neural_networks_for_improving.pdf) | [10.1016/j.sca.2023.100041](https://doi.org/10.1016/j.sca.2023.100041) | publishedVersion | 9 |
| 2023 | [An order fulfilment location planning model for perishable goods supply chains using population density](pdfs/2023/100045_An_order_fulfilment_location_planning_model_for_perishable_goods_supply_chains_using_.pdf) | [10.1016/j.sca.2023.100045](https://doi.org/10.1016/j.sca.2023.100045) | publishedVersion | 21 |
| 2023 | [Exploitation of material consolidation trade-offs in multi-tier complex supply networks](pdfs/2023/100050_Exploitation_of_material_consolidation_trade_offs_in_multi_tier_complex_supply_networ.pdf) | [10.1016/j.sca.2023.100050](https://doi.org/10.1016/j.sca.2023.100050) | publishedVersion | 11 |
| 2024 | [A blockchain-based dynamic energy pricing model for supply chain resiliency using machine learning](pdfs/2024/100066_A_blockchain_based_dynamic_energy_pricing_model_for_supply_chain_resiliency_using_mac.pdf) | [10.1016/j.sca.2024.100066](https://doi.org/10.1016/j.sca.2024.100066) | publishedVersion | 8 |
| 2024 | [A bibliometric analysis of data-driven technologies in digital supply chains](pdfs/2024/100067_A_bibliometric_analysis_of_data_driven_technologies_in_digital_supply_chains.pdf) | [10.1016/j.sca.2024.100067](https://doi.org/10.1016/j.sca.2024.100067) | publishedVersion | 23 |
| 2024 | [Unlocking the potential of digital twins in supply chains: A systematic review](pdfs/2024/100075_Unlocking_the_potential_of_digital_twins_in_supply_chains_A_systematic_review.pdf) | [10.1016/j.sca.2024.100075](https://doi.org/10.1016/j.sca.2024.100075) | publishedVersion | 24 |
| 2024 | [An integrated supply chain network design for advanced air mobility aircraft manufacturing using stochastic optimization](pdfs/2024/100083_An_integrated_supply_chain_network_design_for_advanced_air_mobility_aircraft_manufact.pdf) | [10.1016/j.sca.2024.100083](https://doi.org/10.1016/j.sca.2024.100083) | submittedVersion | 49 |
| 2025 | [A conceptual digital twin framework for supply chain recovery and resilience](pdfs/2025/100091_A_conceptual_digital_twin_framework_for_supply_chain_recovery_and_resilience.pdf) | [10.1016/j.sca.2024.100091](https://doi.org/10.1016/j.sca.2024.100091) | publishedVersion | 20 |
| 2025 | [Optimization-based model of a circular supply chain for coffee waste](pdfs/2025/100126_Optimization_based_model_of_a_circular_supply_chain_for_coffee_waste.pdf) | [10.1016/j.sca.2025.100126](https://doi.org/10.1016/j.sca.2025.100126) | publishedVersion | 19 |
| 2025 | [A game-theoretic framework for optimizing supply chain coordination and production](pdfs/2025/100139_A_game_theoretic_framework_for_optimizing_supply_chain_coordination_and_production.pdf) | [10.1016/j.sca.2025.100139](https://doi.org/10.1016/j.sca.2025.100139) | publishedVersion | 25 |
| 2025 | [A comparative study of multi-algorithm optimization for inventory analytics in supply chains](pdfs/2025/100154_A_comparative_study_of_multi_algorithm_optimization_for_inventory_analytics_in_supply.pdf) | [10.1016/j.sca.2025.100154](https://doi.org/10.1016/j.sca.2025.100154) | publishedVersion | 22 |
| 2026 | [A machine learning and evolutionary optimization framework for carbon-aware supply chain routing](pdfs/2026/100182_A_machine_learning_and_evolutionary_optimization_framework_for_carbon_aware_supply_ch.pdf) | [10.1016/j.sca.2025.100182](https://doi.org/10.1016/j.sca.2025.100182) | publishedVersion | 14 |
| 2026 | [An integrated modeling approach to sustainable transport in forest biomass supply chains](pdfs/2026/100203_An_integrated_modeling_approach_to_sustainable_transport_in_forest_biomass_supply_cha.pdf) | [10.1016/j.sca.2026.100203](https://doi.org/10.1016/j.sca.2026.100203) | publishedVersion | 19 |
| 2026 | [An analytical study on integrating Artificial Intelligence and Industry 4.0 technologies for supply chain resilience and sustainability](pdfs/2026/100209_An_analytical_study_on_integrating_Artificial_Intelligence_and_Industry_4_0_technolog.pdf) | [10.1016/j.sca.2026.100209](https://doi.org/10.1016/j.sca.2026.100209) | publishedVersion | 15 |
| 2026 | [A topological analytics framework for preserving decision structure in supply chain inventory classification](pdfs/2026/100229_A_topological_analytics_framework_for_preserving_decision_structure_in_supply_chain_i.pdf) | [10.1016/j.sca.2026.100229](https://doi.org/10.1016/j.sca.2026.100229) | publishedVersion | 20 |
