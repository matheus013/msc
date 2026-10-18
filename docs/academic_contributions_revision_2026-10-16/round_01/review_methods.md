# Revisão independente — método e evidência

2026-10-16T21:00:18.233586+00:00

Aprovado no escopo metodológico com um reparo menor de precisão. Zero CRITICAL/MAJOR. Isso não constitui prontidão científica para submissão.

## Achado menor MET-01

A nova formulação de seleção contextual sob critérios operacionais e referência comuns pode sugerir que as restrições de serviço têm o mesmo escopo entre abordagens. O corpo distingue meta média global, média por perfil e restrição individual prevista; as instâncias publicadas não estão integralmente harmonizadas.

Locais: `manuscript_ptbr.tex:37` e `capitulos/introducao.tex:69`.

A formulação é programática e as diferenças permanecem explicitadas na metodologia, limitações e continuidade; não há novo resultado nem atribuição causal direta ao algoritmo.

Reparo proposto: Manter contratos de informação e referência explícitos, mas trocar critérios comuns por critérios e escopo da restrição de serviço declarados. Exemplo: integrar o PSE por perfil e por estimação com contratos de informação, referência e escopo da restrição de serviço explícitos.

## Contribuições e evidência

- **Metodologia modular e rastreável para apoio à escolha**: Cinco etapas, contratos, substituição de controladores e seletores, referência e intervenção; contratos completos aparecem como entrega pretendida. Limites: Rastreabilidade parcial, portabilidade de outros módulos e adoção gerencial não validadas.
- **Duas abordagens PSE**: Perfil derivado avaliado na Bahia e M5; estimação por classe e consequências no piloto M5. Limites: Perfil informado externamente admitido, não validado na execução; consequências e classe são variantes da estimação.
- **Diagnóstico de limitações**: 90 casos M5: portfólio retrospectivo, MILP com informação perfeita, demanda zero separada e critérios condicionais. Limites: Sem causalidade no campo ou garantia de antecipar inviabilidade; ótimo condicional não implantável.
- **Aplicações complementares**: Bahia 145 séries/290 casos/4350 ajustes; M5 piloto 30 séries/90 casos/1350 ajustes. Limites: Bases separadas, vendas e custos de cenário, calibrações distintas, família fixa sete ciclos, piloto desenvolvimento.

## Pendências científicas declaradas, preservadas nesta integração

- Comparação empírica com integrações próximas da literatura e delimitação da originalidade.
- Reserva independente e harmonização das restrições/calibração; resultados do piloto não são confirmação.
- Sensibilidades, entrega de parâmetros finais/trajetórias completas, revisão sequencial e validação humana.
- Margens contextuais e desenho próprios se houver pretensão de não inferioridade formal.

## Auditoria independente e alcance

Rerun independente dos padrões do gate mecânico nas linhas acrescentadas ao diff, com os padrões extraídos sem executar o script do escritor.

- M1_em_dash: 0
- M11_passive_EN: 0
- M2_antithesis: 0
- M3_M4_M8_M9: 0
- M6_openers: 0
- M5_M16_words: 1
- M10_hype: 0
- M17_fancy: 0
- M18_empty: 0
- M12_wordiness: 0
- M13_qualifiers: 0
- M14_adverbs: 0
- M15_exclamation: 0
- precision_pairs: 0
- PT_passive_candidates: 0
- PT_promotional: 0

Uma ocorrência de utiliz em português factual na linha herdada A avaliação utiliza sete políticas; falso positivo do filtro inglês. Demais categorias zero nas linhas adicionadas.

Inspeção dos diffs e integrity_checks.json preserva os números e os fragmentos; não foi feita auditoria independente dos bancos de origem nesta rodada.

Nenhum código científico, controlador, resultado ou configuração de execução foi modificado. Não houve simulação, treinamento nem reinício. Fontes revisadas e SHA-256 constam do JSON associado. Inspeção visual e compilação ficam com o coordenador.
