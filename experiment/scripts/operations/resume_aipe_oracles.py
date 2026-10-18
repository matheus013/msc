"""Resume only hindsight evaluation after an audited oracle implementation fix.

Preserves the original training plan/snapshots and frozen selector predictions.
Only oracles.py may differ from the original plan; controller changes require
a new experiment. Cache is archived before recomputing under the new code.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/scripts/analysis'))
import run_aipe_pipeline as pipeline


def resume(run):
    if not run.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid run')
    output=pipeline.BASE/run
    planned=json.loads((output/'plan.json').read_text(encoding='utf-8'))
    hashes=planned['code_sha256']
    allowed=next(name for name in hashes if Path(name).as_posix()==
        'experiment/src/simulation/research/oracles.py')
    for name,digest in hashes.items():
        if pipeline.queue.sha256(output/'code_snapshot'/name)!=digest:
            raise ValueError('Original code snapshot changed: '+name)
        if name!=allowed and pipeline.queue.sha256(ROOT/name)!=digest:
            raise ValueError('Change outside oracle evaluation: '+name)
    for meta in planned['panel_metadata']:
        if pipeline.queue.sha256(output/meta['panel'])!=meta['panel_sha256']:
            raise ValueError('Frozen panel changed')
    if (output/'cancelled_by_user.json').exists():
        raise ValueError('Cancelled run')
    with sqlite3.connect((output/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        count=db.execute('SELECT count(*) FROM fits').fetchone()[0]
    if count!=planned['planned_fits']:
        raise ValueError('Controller fits are incomplete')
    selection=output/'selection'
    frozen=[selection/name for name in ('case_metrics.csv','predictions_frozen_before_scoring.csv',
        'summary.csv','protocol_before_predictions.json','inner_validation.json')]
    before={str(p.relative_to(output)):pipeline.queue.sha256(p) for p in frozen}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    revision=output/'evaluation_revisions'/stamp
    with ExitStack() as stack:
        for path in (pipeline.BASE/'aipe_active_queue.lock',output/'queue.lock'):
            lock=stack.enter_context(path.open('a+b'))
            lock.write(b'0');lock.flush();lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        revision.mkdir(parents=True)
        shutil.copy2(output/'status.json',revision/'prior_status.json')
        shutil.copy2(ROOT/allowed,revision/'oracles.py')
        cache=selection/'oracle_cache.sqlite'
        with sqlite3.connect(cache) as db:
            with sqlite3.connect(revision/'prior_oracle_cache.sqlite') as backup:
                db.backup(backup)
            with db:db.execute('DELETE FROM plans')
        audit={'reason':'Reconstruct order quantities using binary activation to avoid numerical residual fixed costs',
               'original_oracle_sha256':hashes[allowed],
               'new_oracle_sha256':pipeline.queue.sha256(ROOT/allowed),
               'frozen_selection_sha256':before,'controller_fits':count,
               'recovery_script_sha256':pipeline.queue.sha256(Path(__file__))}
        pipeline.queue.write_json(revision/'audit.json',audit)
        status={'state':'controllers_completed','dataset':planned['dataset'],
                'completed_fits':count,'planned_fits':count,'execution':planned['execution'],
                'evaluation_revision':stamp,'pid':__import__('os').getpid()}
        pipeline.queue.write_json(output/'status.json',status)
        try:
            from simulation.research.evaluation import evaluate_oracles
            import pandas as pd
            evaluate_oracles(pipeline.queue.cases(planned['panel_metadata'],output),
                pd.read_csv(output/'kpis.csv',dtype={k:str for k in pipeline.queue.KEYS}),selection)
            if any(pipeline.queue.sha256(output/name)!=digest for name,digest in before.items()):
                raise ValueError('Frozen selection changed during evaluation recovery')
            status['state']='completed'
        except BaseException as exc:
            status.update(state='evaluation_failed',error=repr(exc))
            pipeline.queue.write_json(output/'status.json',status)
            raise
        pipeline.queue.write_json(output/'status.json',status)
        print(json.dumps(status,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    resume(parser.parse_args().run)
