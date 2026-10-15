# Backup dos dados no Google Drive

Destino autorizado: [pasta fornecida pelo autor](https://drive.google.com/drive/folders/1GHk0TAqp4rR310Nro1N4qaodUR6B1JQd), subpasta [msc-backup](https://drive.google.com/drive/folders/1L9gTiQnQydPK239YzpFH05J_Hgpcq3OH).

Código, configurações reproduzíveis, textos e PDFs pertencem ao GitHub. Dados brutos, preparados, matrizes, bancos e modelos permanecem locais e são copiados para o Drive. A exclusão do Git não apaga os arquivos locais. Commit novo não remove dados de commits antigos: não foi reescrito o histórico do repositório.

## Funcionamento

O script `experiment/scripts/operations/drive_backup.py` usa apenas a biblioteca padrão Python. Ele prepara uma cópia completa inicial e, depois, incrementos dos arquivos alterados. A execução é local, com prioridade reduzida no Windows e trava para impedir preparações concorrentes.

O backup abrange `data/`, `experiment/data/` e arquivos de dados tabulares/modelos encontrados em `docs/`. Exclui seu próprio diretório de trabalho, arquivos temporários, travas, logs e arquivos auxiliares WAL/SHM. Os bancos SQLite são copiados pela API de backup online e passam por verificação de integridade. Assim, o checkpoint ativo do M5 pode ser preservado sem interromper o experimento.

Arquivos pequenos, como status, são capturados em memória após conferência de estabilidade. Arquivos maiores precisam manter tamanho e data durante a leitura; se isso não ocorrer, a preparação falha e o snapshot não é confirmado. O backup é consistente por arquivo/banco, não uma transação única entre todos os arquivos da pesquisa.

O arquivo tar.gz é dividido em partes de 32 MiB. Cada parte possui SHA-256 e MD5 no manifesto. O agente envia as partes pelo conector Google Drive e registra o ID confirmado de cada upload. O manifesto é enviado por último. Só então o snapshot é marcado como completo e passa a ser o pai do próximo incremento. Falhas de upload podem ser retomadas pelas partes pendentes. Os arquivos existentes na pasta original do Drive não são alterados.

Arquivos removidos localmente são registrados como exclusões; versões anteriores no Drive permanecem preservadas. Não há exclusão remota automática, espelhamento destrutivo ou retenção que remova cópias antigas. O controle local e os manifests com nomes dos dados ficam fora do Git.

## Comandos locais

Na raiz do projeto:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/drive_backup.py prepare
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/drive_backup.py plan --limit 8
```

`prepare` retorna arquivos a enviar. O conector deve receber o caminho absoluto, nome e pasta exatamente como aparecem no plano. Após cada upload confirmado:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/drive_backup.py ack --name NOME_DA_PARTE --drive-id ID_RETORNADO_PELO_DRIVE
```

Executar `ack` sem upload confirmado invalidaria o controle: não fazer isso. Se o plano indicar `upload_manifest`, enviar o JSON indicado e confirmar da mesma forma. `idle` indica que não há snapshot pendente; não significa, sozinho, que todos os dados atuais já foram copiados.

## Restauração

1. Baixar o manifesto da cópia completa e todos os seus fragmentos, pelos IDs ou nomes registrados.
2. Restaurar em uma pasta vazia, fora do projeto em uso:

```powershell
& experiment/.venv/Scripts/python.exe experiment/scripts/operations/drive_backup.py restore --manifest CAMINHO_MANIFESTO --parts-dir PASTA_DAS_PARTES --destination PASTA_VAZIA
```

3. O comando confere SHA-256 das partes e dos arquivos, rejeita caminhos externos à pasta de destino e não sobrescreve arquivos existentes.
4. Para incrementos, seguir a cadeia `parent` até a cópia completa, restaurar cada incremento em uma pasta vazia própria e aplicar os arquivos em ordem. As exclusões registradas exigem decisão explícita; não são aplicadas automaticamente.

O Drive não substitui a verificação de restauração. A suíte `experiment/tests/test_drive_backup.py` cobre restauração, rejeição de fragmento alterado, cópia de SQLite com WAL ativo, cadeia incremental e alteração entre inventário e leitura. Não foram armazenadas credenciais Google no repositório.

## Agendamento

A rotina recorrente deste chat prepara o snapshot, envia as partes e confirma o manifesto. Ela depende do computador disponível, acesso à pasta local, conexão com a internet e conector Google Drive autenticado. Interrupção não deve ser reportada como backup concluído. O estado durável fica em `experiment/conf/local/drive_backup_state.json`; os pacotes locais ficam em `experiment/data/drive_backup/`.

## Fontes técnicas

- [Python: API de backup do SQLite](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup).
- [Python: tarfile e extração de arquivos](https://docs.python.org/3/library/tarfile.html).

Status real da primeira cópia e referência da rotina agendada são registrados em `BACKUP_STATUS.md` após confirmação dos uploads.
