# Avaliação equivalente com operações vetorizadas

Os 61 arquivos científicos do plano M5 permanecem byte a byte iguais ao plano congelado. A implementação adicional fica em `experiment/scripts/analysis/final_selector_fast.py` e substitui somente operações de agrupamento, ordenação, associação a perfis e consulta de métricas. Não altera modelos, hiperparâmetros, amostras, ordem de treino, validação, critérios ou margens. O frontend `final_postprocess_fast.py` aplica essas substituições exclusivamente na memória do seu próprio processo.

## Verificações concluídas

- `fast_equivalence.json`: 60 comparações exatas das operações de rótulo, decisão e escolha de margem, em todas as origens das matrizes reais Bahia e M5. Hashes das matrizes vinculados à auditoria inicial.
- `fast_semantic_checks.json`: empates, atendimento indefinido, tolerância, duplicidade por política com preservação da primeira linha original, margens infinita/NaN/negativa; metadados, rótulos, ordem dos dados e parâmetros dos estimadores idênticos. Estimadores substituídos por stubs nessa verificação: nenhum ajuste adicional.
- Teste de trava: segunda aquisição simultânea recusada, aquisição permitida após liberar.
- Estado já concluído: retorno sem repetir a avaliação, verificado com a Bahia.

No M5, as operações medidas somaram 288,59 segundos na versão original e 4,07 segundos na versão vetorizada, aproximadamente 70,9 vezes menos tempo. Para escolha da margem, as razões foram 26,9, 63,9 e 76,2 nas origens 50, 64 e 78. Esses números medem operações de processamento, sob carga compartilhada; não são aceleração do treinamento dos controladores nem previsão garantida do tempo total. Na Bahia, o caso artificial de um grupo por loja com poucas séries não acelerou a seleção de margem; não há alegação de ganho universal.

## Ativação explícita

O processo anterior precisa estar encerrado antes de chamar:

```powershell
experiment/.venv/Scripts/python.exe experiment/scripts/analysis/final_postprocess_fast.py --dataset M5 --run aipe_m5_2408_unified_2026-10-15 --former-pid 29964
```

O frontend recusa PID anterior ainda existente, provas de equivalência ausentes/desatualizadas, matriz alterada, código científico diferente do plano e travas já mantidas por outra execução. Mantém as travas global, da fila e de avaliação até terminar. Publica estado com novo PID, um processo, uma thread e prioridade abaixo do normal; preserva a configuração original dos controladores separadamente. Em falha, grava `evaluation_failed`, erro e contagens preservadas. Captura status anterior, hashes do plano/matriz/scripts e provas em `execution_revisions/fast_postprocess_*`. Não executa controladores.
