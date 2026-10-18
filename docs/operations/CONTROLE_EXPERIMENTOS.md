# Controle de concorrência dos experimentos

Abra <http://127.0.0.1:8765/> e use **Processos de CPU → Aplicar** no M5.
O limite pode variar entre 1 e 16 durante a execução. Aumento cria novos
processos; redução aguarda os ajustes em andamento terminarem. Durante essa
transição, o número de ajustes ativos pode exceder temporariamente o novo limite.
Cada processo usa uma thread de treinamento e prioridade reduzida.

O controle altera somente o escalonamento: séries, políticas, sementes e
hiperparâmetros permanecem os do plano congelado. Checkpoints concluídos são
preservados e o processo principal continua sendo o único escritor no banco.
As projeções de tempo continuam aproximadas; mais processos não garantem
aceleração proporcional.

O executor operacional é `experiment/scripts/operations/run_aipe_elastic.py`.
Ele valida os hashes científicos antes de iniciar e registra a configuração em
`elastic_execution.json`, sem modificar o plano original. O comando de retomada
do M5 é:

```powershell
experiment/.venv/Scripts/python.exe experiment/scripts/operations/run_aipe_elastic.py --run aipe_m5_2408_unified_2026-10-15 --workers 12
```

Não inicie outro executor para a mesma fila. A trava global impede execução
simultânea. A migração inicial preservou todos os 10.605 checkpoints e verificou
a integridade do SQLite; somente ajustes sem checkpoint podem ser repetidos.

Arquivos na pasta da execução:

- `runtime_control.json`: limite solicitado, lido durante a execução.
- `runtime_pool.json`: limite reconhecido, ajustes ativos e processos em retirada.
- `runtime_control_history.jsonl`: histórico das solicitações pelo painel.
- `elastic.stdout.log` e `elastic.stderr.log`: saída do executor atual.

O painel aceita alterações somente da própria origem local e apenas para uma
fila elástica ativa. Ele não altera dados, parâmetros científicos ou checkpoints.
O Brasil permanece fora desta execução.

Atualizações do monitoramento tentam novamente quando o Windows bloqueia a
substituição do arquivo. Se o bloqueio persistir, o executor registra um aviso
e continua os treinamentos; a falha da telemetria não encerra a fila. O último
estado exibido pode permanecer desatualizado até a gravação voltar a funcionar.
