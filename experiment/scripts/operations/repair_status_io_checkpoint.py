"""Audited one-time migration for the Windows status-file repair only.

Rejects any scientific code change. Preserves the old manifest and code snapshot.
"""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'experiment/data/08_reporting/research_v2/census_2026-10-14'
RUNNER='experiment/scripts/analysis/run_production_v2.py'


def tree_without_status_writer(source):
    tree=ast.parse(source)
    tree.body=[node for node in tree.body if not(isinstance(node,ast.FunctionDef) and node.name=='write_json')]
    return ast.dump(tree,include_attributes=False)


def main():
    manifest_path=OUTPUT/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    changed=[name for name,digest in manifest['code_sha256'].items()
             if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if [name.replace('\\','/') for name in changed]!=[RUNNER]:raise ValueError(f'Only the status writer may change: {changed}')
    runner_key=changed[0]
    old=OUTPUT/'code_snapshot'/RUNNER
    current=ROOT/RUNNER
    if tree_without_status_writer(old.read_text(encoding='utf-8'))!=tree_without_status_writer(current.read_text(encoding='utf-8')):
        raise ValueError('Scientific runner changed outside write_json; use a new run')
    archive=OUTPUT/'manifest_before_status_io_repair.json'
    if archive.exists():raise FileExistsError('Migration already performed')
    shutil.copy2(manifest_path,archive)
    before=OUTPUT/'code_snapshot_before_status_io_repair'/RUNNER
    before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,before)
    old_hash=manifest['code_sha256'][runner_key]
    new_hash=hashlib.sha256(current.read_bytes()).hexdigest()
    shutil.copy2(current,old)
    with sqlite3.connect(OUTPUT/'results.sqlite') as db:
        completed=db.execute('SELECT count(*) FROM fits').fetchone()[0]
    manifest['code_sha256'][runner_key]=new_hash
    manifest['status_io_repair']={'utc':datetime.now(timezone.utc).isoformat(),
        'old_runner_sha256':old_hash,'new_runner_sha256':new_hash,
        'verified':'AST identical outside write_json; all other scientific source hashes unchanged.',
        'committed_fits_preserved':completed}
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Verified transport-only repair. Preserved completed fits:',completed)


if __name__=='__main__':main()
