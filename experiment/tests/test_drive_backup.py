import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('drive_backup',Path(__file__).resolve().parents[1]/'scripts/operations/drive_backup.py')
backup=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(backup)

class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.original=(backup.ROOT,backup.LOCAL,backup.STATE)
        backup.ROOT=self.root;backup.LOCAL=self.root/'experiment/data/drive_backup';backup.STATE=self.root/'experiment/conf/local/state.json'
        (self.root/'data').mkdir();(self.root/'data/example.csv').write_text('x,y\n1,2\n')
    def tearDown(self):
        backup.ROOT,backup.LOCAL,backup.STATE=self.original;self.temp.cleanup()
    def make(self):
        # Default arguments bind at definition time; provide this fixture's inventory.
        original=backup.candidates;backup.candidates=lambda:list(original(self.root))
        try:return backup.prepare(1)
        finally:backup.candidates=original
    def test_round_trip_and_manifest_order(self):
        plan=self.make();state=backup.load_state();path=self.root/state['pending']
        with self.assertRaises(ValueError):backup.ack(path.name,'manifestID')
        for item in plan['uploads']:backup.ack(item['file_name'],'partID')
        self.assertEqual(backup.plan()['status'],'upload_manifest')
        restored=self.root/'restore';backup.restore(path,path.parent,restored)
        self.assertEqual((restored/'data/example.csv').read_bytes(),(self.root/'data/example.csv').read_bytes())
        backup.ack(path.name,'manifestID');self.assertEqual(backup.plan()['status'],'idle')
        self.assertEqual(self.make()['status'],'unchanged')
    def test_sqlite_snapshot_includes_committed_wal(self):
        source=self.root/'data/live.sqlite'
        with closing(sqlite3.connect(source)) as db:
            db.execute('pragma journal_mode=wal');db.execute('create table progress(n)');db.execute('insert into progress values (42)');db.commit()
            target=self.root/'snapshot.sqlite';backup.sqlite_snapshot(source,target)
            with closing(sqlite3.connect(target)) as saved:self.assertEqual(saved.execute('select n from progress').fetchone()[0],42)
    def test_tampered_part_is_rejected(self):
        self.make();path=self.root/backup.load_state()['pending'];m=json.loads(path.read_text())
        (path.parent/m['parts'][0]['name']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):backup.restore(path,path.parent,self.root/'restore')
    def test_incremental_preserves_parent(self):
        self.make();path=self.root/backup.load_state()['pending']
        for item in backup.plan()['uploads']:backup.ack(item['file_name'],'partID')
        backup.ack(path.name,'manifestID')
        (self.root/'data/example.csv').write_text('x,y\n3,4\n')
        self.make();m=json.loads((self.root/backup.load_state()['pending']).read_text())
        self.assertEqual(m['kind'],'incremental');self.assertEqual(m['parent']['manifest_drive_id'],'manifestID')
    def test_file_changed_after_inventory_uses_stable_latest_bytes(self):
        original=backup.signature;target=self.root/'data/example.csv';calls=0
        def changing(path):
            nonlocal calls
            result=original(path)
            if path==target:
                calls+=1
                if calls==1:target.write_text('updated safely before archive read\n')
            return result
        with patch.object(backup,'signature',side_effect=changing):self.make()
        path=self.root/backup.load_state()['pending'];restored=self.root/'restore'
        backup.restore(path,path.parent,restored)
        self.assertEqual((restored/'data/example.csv').read_bytes(),target.read_bytes())

if __name__=='__main__':unittest.main()
