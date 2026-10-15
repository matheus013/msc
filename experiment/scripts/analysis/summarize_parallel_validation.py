"""Build a reproducible runtime report from completed, isolated trials."""
from pathlib import Path
import json
from benchmark_parallel_fits import ROOT, compare_results, write_json
import pandas as pd


def main():
    base=ROOT/'experiment/data/08_reporting/performance'
    reference=base/'parallel_validation_2026-10-14'
    background=base/'background_validation_2026-10-14'
    cpu_trials=[json.loads(p.read_text()) for p in sorted(reference.glob('*_cpu_*.json'))]
    baseline=next(t for t in cpu_trials if t['summary']['phase']=='screen' and t['summary']['workers']==1)
    full_baseline=next(t for t in cpu_trials if t['summary']['phase']=='full' and t['summary']['workers']==1)
    checks=[]
    for t in cpu_trials:
        ref=baseline if t['summary']['phase']=='screen' else full_baseline
        checks.append({'phase':t['summary']['phase'],'workers':t['summary']['workers'],
                       'maximum_metric_difference':compare_results(ref['results'],t['results'])})
    write_json(reference/'manifest.json',{'status':'completed CPU scaling and full-budget confirmation',
        'cpu_equivalence_checks':checks,'production_queue':'not resumed',
        'gpu_stages':'Replaced by shorter matched trials in background_validation_2026-10-14; optional original GPU sweep was stopped before completion.',
        'scope':'One series per base, BA origin 17 and M5 origin 50, DQN/PPO/GA-DQN. Four seeds at 2,048 transitions, two seeds at 20,000, rounded to episodes.',
        'limitations':['One timing repetition per setting, uncontrolled concurrent user workload.',
                       'Normal priority in CPU scaling; below-normal in the separate background benchmark.',
                       'Metadata corrected: all odd seeds select the same single series per base.']})
    pd.DataFrame([t['summary'] for t in cpu_trials]).to_csv(reference/'throughput.csv',index=False)
    manifest=json.loads((background/'manifest.json').read_text())
    bg_trials=[json.loads(p.read_text()) for p in sorted(background.glob('*_cpu_*.json'))+sorted(background.glob('*_cuda_*.json'))]
    def table(trials):
        lines=['| Dispositivo | Processos | Treinos | Inicialização (s) | Lote após inicialização (s) | Treinos/min | RAM livre mínima (GB) |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for t in sorted(trials,key=lambda t:(t['summary']['device'],t['summary']['workers'])):
            r=t['summary'];lines.append(f"| {r['device']} | {r['workers']} | {r['jobs']} | {r['startup_seconds']:.2f} | {r['steady_seconds']:.2f} | {r['fits_per_minute']:.2f} | {r['min_available_ram_mb']/1024:.2f} |")
        return '\n'.join(lines)
    screen=[t for t in cpu_trials if t['summary']['phase']=='screen']
    full=[t for t in cpu_trials+bg_trials if t['summary']['phase']=='full']
    short_bg=[t for t in bg_trials if t['summary']['phase']=='screen']
    cpu=next(t for t in short_bg if t['summary']['device']=='cpu' and t['summary']['workers']==1)
    cuda=next(t for t in short_bg if t['summary']['device']=='cuda' and t['summary']['workers']==1)
    pairs=[]
    for policy in ['DQN','PPO','GA-DQN']:
        a=[r['fit_seconds'] for r in cpu['results'] if r['policy']==policy]
        b=[r['fit_seconds'] for r in cuda['results'] if r['policy']==policy]
        mean_cpu=sum(a)/len(a);mean_cuda=sum(b)/len(b)
        pairs.append(f'| {policy} | {mean_cpu:.3f} | {mean_cuda:.3f} | {mean_cuda/mean_cpu:.2f} |')
    expected_priority=16384  # Windows BELOW_NORMAL_PRIORITY_CLASS
    assert all(r['process_priority']==expected_priority for t in bg_trials for r in t['results'])
    report=f'''# Validação de CPU, GPU e paralelismo — 14/10/2026

Perfil solicitado: rodar junto com os aplicativos do usuário. A fila censitária permanece parada; nenhum resultado científico do artigo foi substituído por estes testes.

## Escopo e controle

Uma série real da Bahia (origem 17) e uma série M5 (origem 50), DQN, PPO e GA-DQN. Triagem: quatro sementes e aproximadamente 2.048 transições por treino, 24 treinos por configuração. Confirmação: duas sementes e aproximadamente 20.000 transições por treino, 12 treinos por configuração. Episódios completos arredondam o orçamento para cima. O teste curto do perfil de uso cotidiano usa uma semente, seis treinos idênticos em cada configuração CPU/CUDA.

Cada processo limita Torch/BLAS a uma thread. O cronômetro do lote inclui treinamento, avaliação e comunicação com o processo principal; inicialização e aquecimento são mostrados separadamente. A RAM livre se refere à máquina inteira, incluindo outros aplicativos. Há uma repetição por configuração; a carga externa variou entre elas, portanto diferenças pequenas não definem um ótimo universal.

## Escala da CPU, prioridade normal

{table(screen)}

## Orçamento completo de treinamento

As linhas de 1 e 12 processos usam prioridade normal. A linha de 8 processos usa prioridade reduzida, conforme o perfil solicitado; a diferença entre elas também inclui a variação da carga externa.

{table(full)}

As métricas de validação e teste permaneceram idênticas entre as execuções CPU, incluindo a confirmação com prioridade reduzida. Verificações e tempos individuais estão nos manifestos e JSONs das duas pastas de medição.

## Uso cotidiano: CPU e GPU com prioridade reduzida

A prioridade Windows foi conferida em todos os processos executores (BELOW_NORMAL_PRIORITY_CLASS). Isso favorece os aplicativos ativos na disputa por CPU, sem garantir limite de utilização ou de latência da interface. Não foram fechados aplicativos do usuário.

{table(short_bg)}

Tempos médios apenas de treinamento, um processo em cada dispositivo, nos mesmos seis casos:

| Política | CPU (s) | GPU (s) | GPU/CPU |
|---|---:|---:|---:|
{chr(10).join(pairs)}

O código faz inferências e atualizações pequenas, com conversões frequentes de arrays para tensores e extrações de escalares (`.item()`). Na GPU essas operações exigem transferências e sincronizações; o diagnóstico anterior de DQN já apontava o replay como gargalo. Os tempos medidos são compatíveis com esse custo predominando sobre o cálculo das redes pequenas. Processos CUDA adicionais também duplicam contextos e aumentam o consumo de memória; isso não equivale a treinar várias redes em um único lote vetorizado na GPU.

CPU e CUDA podem aprender políticas diferentes por usarem fluxos aleatórios e operações numéricas diferentes; igualdade de eficácia não é alegada. Os números curtos não são uma previsão de duração do censo.

## Configuração implementada

O executor de produção usa CPU, 8 processos e prioridade reduzida por padrão, com uma thread por processo e no máximo 16 tarefas pendentes. Há uma única conexão escritora SQLite, com transação por treino, sementes independentes, retomada apenas dos treinos não confirmados e registro do dispositivo/configuração no manifesto. Alterar código ou perfil exige outra identificação de rodada para não misturar resultados. A ablação DQN continua separada do portfólio principal.

O launcher aceita `-Workers`, `-Device` e `-Priority`; o script Python aceita `--workers`, `--device` e `--priority`. Preparar esses padrões não reiniciou a execução nacional. O perfil de 16 processos reduziu a folga de RAM na triagem e não foi adotado como padrão para uso simultâneo.

Trinta testes passaram, incluindo transações, retomada, correspondência dos resultados à tarefa e consumo limitado da fila. Os testes reais entre processos também verificaram a invariância das métricas com as mesmas sementes.

Fontes locais: `experiment/data/08_reporting/performance/parallel_validation_2026-10-14/` e `experiment/data/08_reporting/performance/background_validation_2026-10-14/`. Esta é uma medição de execução, não uma comparação final da qualidade das políticas.
'''
    path=ROOT/'docs/supply_chain_analytics_article/DESEMPENHO_CPU_GPU.md'
    path.write_text(report,encoding='utf-8')
    print(path)
    print(pd.DataFrame([t['summary'] for t in bg_trials]).to_string(index=False))


if __name__=='__main__':main()
