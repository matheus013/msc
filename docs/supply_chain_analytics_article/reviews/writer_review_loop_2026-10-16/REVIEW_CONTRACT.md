# Contrato dos três revisores

Todos os revisores leem a mesma versão congelada da rodada. A avaliação é interna e assistida por modelos; independência entre tarefas não equivale a revisão humana ou diversidade de modelos. Os pareceres não devem ser consultados entre si antes de concluídos.

## Revisor 1: contribuição e argumento

Pergunta central: o leitor identifica o problema, a contribuição metodológica, o mecanismo e a razão para investigá-lo? Examinar posicionamento frente aos antecedentes citados, consistência entre resumo/introdução/método/conclusão, papel das bases e alcance da modularidade. Não transformar ausência de novidade comprovada em afirmação de prioridade. Distinguir contribuição proposta de benefício demonstrado.

## Revisor 2: método e evidência

Pergunta central: o texto permite interpretar corretamente cada comparação? Examinar disponibilidade de informação, calibração/seleção/teste, serviço agregado/individual, sementes, unidade de análise, suporte, recuo, oráculos, incerteza e números. Comparar com artefatos existentes quando necessário, sem treinar nada nem modificar o código. Explicitar incertezas não resolvidas.

## Revisor 3: clareza e apresentação

Pergunta central: um leitor da área, sem o histórico deste chat, acompanha o artigo com esforço razoável? Examinar fluxo entre parágrafos, primeira definição de termos, concisão, repetições, títulos, notação, legendas e relação figura/texto. Revisar o PDF compilado e as seções alteradas. Não remover uma restrição científica necessária para tornar o texto mais persuasivo.

## Formato exigido

Em review_<papel>.md, registrar: versão/hash lido, decisão editorial, situação científica para submissão, forças, achados numerados por gravidade, localização, evidência, reparo e critério de verificação. Concluir com pendências científicas/autorais que não se resolvem apenas por escrita.

Em review_<papel>.json:

```json
{
  "round": 1,
  "reviewer": "contribution|methods|editorial",
  "source_sha256": "hash real do manuscrito congelado",
  "editorial_decision": "approve|minor_revision|major_revision",
  "scientific_submission_ready": false,
  "issues": [
    {
      "id": "identificador estável",
      "severity": "critical|major|minor",
      "location": "arquivo/seção/linha",
      "evidence": "o que o texto ou artefato efetivamente mostra",
      "requested_change": "reparo verificável",
      "kind": "writing|claim_scope|experiment|author_decision",
      "status": "open|resolved|deferred_with_reason"
    }
  ]
}
```

## Critério de retorno ao escritor

Critical/major de escrita ou escopo exigem revisão. Minor pode exigir correção local; preferência estilística opcional não impede fechamento. Problema experimental não resolvido permanece aberto no registro científico; ajustar a alegação é aceitável somente quando elimina o problema real da alegação e mantém a limitação explícita. Não baixar gravidade apenas para alcançar convergência.

Na rodada seguinte, verificar regressão dos achados anteriores e os novos trechos. Reabrir questão resolvida exige evidência nova ou falha verificável. Se o artigo continuar insuficiente, solicitar nova rodada com reparos específicos, não uma ordem vaga de "melhorar mais".
