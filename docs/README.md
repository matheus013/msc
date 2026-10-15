# Mestrado — AIPE

O [artigo atual](supply_chain_analytics_article/README.md) e a [dissertação](master_proposal/README.md) apresentam a avaliação temporal da Bahia como evidência principal. A qualificação permanece preservada como antecedente: [fontes e instruções](qualification_presentation/README.md) e [PDF](qualification_presentation/main.pdf).

O [novo artigo para Supply Chain Analytics](supply_chain_analytics_article/README.md) contém o manuscrito atual em português, a versão inglesa anterior, perfis de agentes, pareceres e a documentação da evidência. A auditoria identificou diferenças entre versões do benchmark; seus resultados não substituem silenciosamente os artefatos da qualificação.

## Estrutura

```text
data/                  Fontes brutas, locais e não versionadas
experiment/            Projeto Python integrado, configurações e experimentos
  src/                 Núcleo compartilhado, pipelines, preparação e relatórios
  scripts/             Utilitários acessados pela entrada única run.py
  conf/                Parâmetros Kedro e configuração do experimento PB
  data/                Dados preparados, modelos e resultados locais
  tests/               Validações de ingestão, simulação e GPU
docs/                  Produção acadêmica, literatura e materiais históricos
  qualification_presentation/  Qualificação: referência atual dos resultados
  master_proposal/             Texto da proposta/dissertação em LaTeX
  references/                  Artigos PDF e notas de literatura
  policy_evolution_presentation/  Material técnico sobre as políticas
  archive/                     SBPO descontinuado e suas versões anteriores
```

## Execução

As instruções de instalação, GPU e comandos ficam em [experiment/README.md](../experiment/README.md). A entrada única é `python experiment/run.py`, executada da raiz, ou `python run.py` dentro de `experiment/`.

## Resultados e preservação

A apresentação da qualificação e seus artefatos existentes foram preservados. O SBPO foi arquivado e não é mais uma entrega ativa. Artigos, figuras e apresentações em PDF não devem ser apagados.

Os dados brutos ficam apenas em `data/`; dados preparados, modelos e resultados ficam em `experiment/data/`. Nenhum dado bruto foi reprocessado nesta reorganização. As mudanças no código valem para execuções futuras e não alteram os resultados já apresentados.

Consulte [ORGANIZACAO.md](ORGANIZACAO.md) para o registro da migração.

Código, configurações, documentação e PDFs são versionados no GitHub. Dados brutos, tabelas de dados, bancos e modelos ficam fora do Git, com cópia versionada no [Google Drive](operations/BACKUP_GOOGLE_DRIVE.md).
