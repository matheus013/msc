# Módulo opcional de contexto do AIPE

Implementação isolada do executor científico: pacote `aipe_context`, comando operacional, configuração desabilitada e documentação de uso. O gestor pode fixar, submeter à seleção ou excluir atributos. A seleção interna exige ganho de custo sob tolerâncias explícitas de atendimento e limita o número de atributos e ajustes. A recomendação protegida mantém o PSE básico quando não há ganho interno admissível.

Foram aprovados 27 testes sintéticos, incluindo disponibilidade temporal, seleção em partições internas, isolamento de grupos externos, congelamento anterior à leitura do teste, persistência de modelos e bloqueio enquanto as rodadas configuradas estão ativas. Nenhuma simulação, ajuste de controlador ou avaliação com contexto real foi iniciada.

Na verificação, Bahia estava concluída (4.350 ajustes) e M5 em execução (51.745 de 79.890). As 61 fontes congeladas do M5 coincidiam com os arquivos atuais. Na Bahia, a única diferença era uma revisão anterior de `oracles.py`, registrada separadamente; as três dependências da reconstrução de atributos coincidiam com os planos de ambas as bases. Nenhuma dessas fontes foi modificada nesta implementação.

O módulo permanece desligado. A avaliação empírica depende da conclusão das rodadas e da comprovação das datas de disponibilidade dos atributos e rótulos. Os atributos atuais não foram declarados historicamente válidos por presunção. O código permite adicionar novas fontes; não demonstra ainda que contexto melhora o AIPE.

Detalhes, comandos e contratos: `docs/operations/MODULO_CONTEXTO_OPCIONAL_AIPE.md`. Evidências e hashes: `verification.json` nesta pasta.
