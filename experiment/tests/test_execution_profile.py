import importlib.util
from pathlib import Path
import sys
import pytest


def scoped_module():
    script=Path(__file__).resolve().parents[1]/'scripts/analysis/run_scoped_production.py'
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location('profile_migration_queue',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_operational_profile_migration_keeps_scientific_hashes_and_history(tmp_path):
    queue=scoped_module();script=Path(queue.__file__);key=str(script.relative_to(queue.ROOT))
    old=tmp_path/'code_snapshot'/key;old.parent.mkdir(parents=True)
    old.write_text(script.read_text(encoding='utf-8')+'\n# Previous operational version\n',encoding='utf-8')
    old_hash=queue.queue.sha256(old)
    previous_execution={'workers':16,'priority':'normal','device':'cpu'}
    execution={'workers':8,'priority':'below-normal','device':'cpu'}
    manifest={'execution':previous_execution,'code_sha256':{key:old_hash,'scientific.py':'unchanged'}}
    signature={key:queue.queue.sha256(script),'scientific.py':'unchanged'}
    queue.update_execution_profile(manifest,signature,execution,tmp_path)
    assert manifest['execution']==execution
    assert manifest['code_sha256']['scientific.py']=='unchanged'
    assert manifest['execution_history'][0]['previous_execution']==previous_execution
    assert (tmp_path/'operational_runner_history'/old_hash/'run_scoped_production.py').exists()


def test_profile_migration_rejects_scientific_source_changes(tmp_path):
    queue=scoped_module()
    with pytest.raises(ValueError,match='Scientific code changed'):
        queue.update_execution_profile({'execution':{},'code_sha256':{'scientific.py':'old'}},
                                       {'scientific.py':'new'},{'workers':8},tmp_path)
