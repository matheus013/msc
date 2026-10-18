"""Explicit opt-in equivalent bookkeeping frontend; no controller execution.

Only run after the previous evaluator has stopped. Scientific source files are
verified against the frozen run plan and remain unmodified. Models and their
parameters/row ordering are unchanged; only equivalent pandas bookkeeping is
replaced in this isolated interpreter.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,shutil
from contextlib import ExitStack
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiment/src'));sys.path.insert(0,str(Path(__file__).parent))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 temp=path.with_name(path.name+f'.fast.{os.getpid()}.tmp')
 temp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temp,path)
def acquire(path):
 import msvcrt
 handle=path.open('a+b')
 try:
  if handle.tell()==0:handle.write(b'0');handle.flush()
  handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
  return handle
 except BaseException:handle.close();raise
def main():
 import psutil
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['M5','BA'],required=True);parser.add_argument('--run',required=True);parser.add_argument('--former-pid',type=int,required=True);args=parser.parse_args()
 if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run')
 run=ROOT/'experiment/data/08_reporting/research_v2'/args.run
 status=json.loads((run/'status.json').read_text())
 if status.get('state')=='completed':print('Evaluation already completed; nothing repeated.');return
 if psutil.pid_exists(args.former_pid):raise ValueError('Former evaluator still exists; do not duplicate evaluation')
 plan=json.loads((run/'plan.json').read_text());assert plan['dataset']==args.dataset
 differences=[name for name,digest in plan['code_sha256'].items() if sha(ROOT/name)!=digest]
 if differences:raise ValueError('Frozen code differs: '+str(differences))
 if status['completed_fits']!=plan['planned_fits']:raise ValueError('Controller matrix incomplete')
 benchmark=ROOT/'docs/operations/m5_final_2026-10-17/fast_equivalence.json'
 proof=json.loads(benchmark.read_text())
 helper=Path(__file__).with_name('final_selector_fast.py')
 if proof['status']!='all_exact' or proof['source_sha256']!=sha(helper):raise ValueError('Exact equivalence record missing or stale')
 if proof.get('raw_matrices_sha256',{}).get(str((run/'kpis.csv').relative_to(ROOT)))!=sha(run/'kpis.csv'):raise ValueError('Equivalence matrix hashes missing or stale')
 semantic=benchmark.with_name('fast_semantic_checks.json');checks=json.loads(semantic.read_text())
 if checks['status']!='all_exact' or checks['helper_sha256']!=sha(helper):raise ValueError('Additional semantic checks missing or stale')
 # Locks are compatible with the canonical execution lock and add an explicit
 # evaluation lock. They are held until scoring/oracles and provenance finish.
 with ExitStack() as locks:
  for path in [run.parent/'aipe_active_queue.lock',run/'queue.lock',run/'selector_evaluation.lock']:
   handle=acquire(path);locks.callback(handle.close)
  locked_main(args,run,plan,proof,helper,benchmark,semantic,status)
def locked_main(args,run,plan,proof,helper,benchmark,semantic,old_status):
 import psutil
 status=json.loads((run/'status.json').read_text())
 if status.get('state')=='completed':print('Evaluation already completed; nothing repeated.');return
 if status!=old_status:raise ValueError('Execution status changed before locking; inspect before retry')
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 revision=run/'execution_revisions'/datetime.now(timezone.utc).strftime('fast_postprocess_%Y%m%dT%H%M%SZ');revision.mkdir(parents=True,exist_ok=False)
 for source in [Path(__file__),helper,benchmark,semantic]:shutil.copy2(source,revision/source.name)
 audit={'kind':'equivalent_selector_bookkeeping_only','started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'former_pid':args.former_pid,'dataset':args.dataset,'frozen_code_files_verified':len(plan['code_sha256']),'frozen_code_differences':[],'controller_fits_started':0,'model_changes':False,'raw_kpis_sha256':sha(run/'kpis.csv'),'plan_sha256':sha(run/'plan.json'),'old_status':old_status,'frontend_sha256':sha(Path(__file__)),'helper_sha256':sha(helper),'equivalence_sha256':sha(benchmark),'patched_helpers':['pse.predict_loso','pse_consequences.matched_classifier','pse_consequences.decisions','pse_consequences.tune_margin']}
 (revision/'audit_before.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
 import final_selector_fast as fast
 import simulation.research.pse as pse
 import simulation.research.pse_consequences as consequences
 pse.predict_loso=fast.predict_loso_fast
 consequences.matched_classifier=fast.matched_classifier_fast
 consequences.decisions=fast.decisions_fast
 consequences.tune_margin=fast.tune_margin_fast
 # Evaluation imports resolve these functions in this interpreter only.
 import run_aipe_pipeline as pipeline
 active={**old_status,'state':'controllers_completed','phase':'selector_evaluation','pid':os.getpid(),'updated_utc':datetime.now(timezone.utc).isoformat(),'postprocessing_revision':str(revision.relative_to(run)),'controller_execution':old_status.get('execution'),'execution':{'workers':1,'device':'cpu','priority':'below-normal','threads_per_worker':1,'dynamic_workers':False}}
 write(run/'status.json',active)
 start=time.perf_counter()
 try:pipeline.evaluate(run,plan)
 except BaseException as exc:
  write(run/'status.json',{**active,'state':'evaluation_failed','updated_utc':datetime.now(timezone.utc).isoformat(),'error':repr(exc)})
  write(revision/'failure.json',{'error':repr(exc),'elapsed_seconds':time.perf_counter()-start,'controller_fits_started':0})
  raise
 audit['elapsed_seconds']=time.perf_counter()-start;audit['completed_utc']=datetime.now(timezone.utc).isoformat();audit['status']='completed'
 audit['frozen_source_files_still_match']=all(sha(ROOT/name)==digest for name,digest in plan['code_sha256'].items())
 audit['kpis_unchanged']=sha(run/'kpis.csv')==audit['raw_kpis_sha256']
 audit['plan_unchanged']=sha(run/'plan.json')==audit['plan_sha256']
 write(run/'status.json',{**active,'state':'completed','phase':'completed','updated_utc':datetime.now(timezone.utc).isoformat()})
 (revision/'completion.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
