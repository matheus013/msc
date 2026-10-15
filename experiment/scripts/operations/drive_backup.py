"""Incremental, versioned data snapshots for upload through the Drive connector.

No Drive credentials are stored here. The agent uploads plan entries and confirms
their returned IDs with ack. A manifest is uploaded only after every archive part.
SQLite uses its online backup API; other files must remain stable while read.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import time
from datetime import datetime, timezone
from contextlib import closing, contextmanager

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'experiment/data/drive_backup'
STATE=ROOT/'experiment/conf/local/drive_backup_state.json'
FOLDER='1L9gTiQnQydPK239YzpFH05J_Hgpcq3OH'
DATA_SUFFIXES={'.csv','.tsv','.parquet','.arrow','.feather','.pkl','.pickle','.sqlite','.db','.npy','.npz','.joblib','.pt','.pth','.h5','.hdf5','.xlsx','.xls'}

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def load_state():
    return json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {'folder_id':FOLDER,'last':None,'files':{},'pending':None}

@contextmanager
def exclusive_run():
    STATE.parent.mkdir(parents=True,exist_ok=True)
    with (STATE.parent/'drive_backup.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            lock.seek(0)
            if os.name=='nt':msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def candidates(root=ROOT):
    roots=[root/'data',root/'experiment/data']
    for base in roots:
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in {'drive_backup','__pycache__'}]
            for name in sorted(files):
                p=Path(folder)/name
                if p.is_symlink() or name.endswith(('-wal','-shm','.lock','.tmp','.log')):continue
                yield p
    # Data derivatives outside data folders must follow the same backup policy.
    for p in (root/'docs').rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in DATA_SUFFIXES:yield p

def signature(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns]

class PartsWriter:
    def __init__(self,folder,prefix,limit):
        self.folder=folder;self.prefix=prefix;self.limit=limit;self.current=None;self.size=0;self.parts=[]
    def write(self,data):
        length=len(data);view=memoryview(data)
        while view:
            if self.current is None:
                self.path=self.folder/f'{self.prefix}.part{len(self.parts)+1:04d}.tar.gz.chunk'
                self.current=self.path.open('wb');self.digest=hashlib.sha256();self.md5=hashlib.md5();self.size=0
            block=view[:self.limit-self.size];self.current.write(block);self.digest.update(block);self.md5.update(block)
            self.size+=len(block);view=view[len(block):]
            if self.size==self.limit:self.finish_part()
        return length
    def finish_part(self):
        if self.current:
            self.current.close();self.parts.append({'name':self.path.name,'size':self.size,'sha256':self.digest.hexdigest(),'md5':self.md5.hexdigest(),'drive_id':None});self.current=None
    def flush(self):
        if self.current:self.current.flush()
    def close(self):self.finish_part()

class HashReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def read(self,size=-1):
        data=self.stream.read(size);self.digest.update(data);return data

def sqlite_snapshot(source,target,timeout=60):
    started=time.monotonic()
    def progress(status,remaining,total):
        if time.monotonic()-started>timeout:raise TimeoutError('SQLite snapshot exceeded time limit')
    with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True,timeout=5)) as src:
        with closing(sqlite3.connect(target)) as dst:src.backup(dst,pages=256,progress=progress,sleep=.01)
    with closing(sqlite3.connect(target)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('SQLite snapshot failed integrity check')

def prepare(part_mib=32):
    state=load_state()
    if state.get('pending'):return plan()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=LOCAL/stamp;folder.mkdir(parents=True)
    scratch=folder/'sqlite_snapshots';scratch.mkdir()
    cache=state.get('files',{});changed=[];inventory={};deferred=[]
    for p in sorted(set(candidates())):
        rel=p.relative_to(ROOT).as_posix();sig=signature(p);inventory[rel]=sig
        if p.suffix.lower() in {'.sqlite','.db'} or cache.get(rel)!=sig:changed.append((p,rel,sig))
    if not changed:return {'status':'unchanged','files':len(inventory)}
    writer=PartsWriter(folder,'msc-'+stamp,part_mib*1024**2);records=[]
    try:
        with tarfile.open(fileobj=writer,mode='w|gz',compresslevel=1) as archive:
            for index,(p,rel,before) in enumerate(changed):
                # Enumeration may precede this read by minutes on the first copy.
                before=signature(p);inventory[rel]=before
                source=p
                if p.suffix.lower() in {'.sqlite','.db'}:
                    source=scratch/f'{index}.sqlite';sqlite_snapshot(p,source)
                elif before[0]<=2*1024**2:
                    # Capture volatile status/manifest files before streaming the archive.
                    for attempt in range(4):
                        before=signature(p);data=p.read_bytes()
                        if signature(p)==before:break
                        time.sleep(.05)
                    else:raise RuntimeError('Source remained in motion: '+rel)
                    inventory[rel]=before
                    info=tarfile.TarInfo(rel);info.size=len(data);info.mtime=before[1]/1e9
                    archive.addfile(info,io.BytesIO(data))
                    records.append({'path':rel,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'sqlite_online_snapshot':False})
                    continue
                info=archive.gettarinfo(str(source),arcname=rel)
                if not info.isfile():raise ValueError('Only regular files are allowed')
                with source.open('rb') as stream:
                    reader=HashReader(stream);archive.addfile(info,reader)
                if source==p and signature(p)!=before:
                    raise RuntimeError('Source changed while reading: '+rel+'; prepare a fresh snapshot')
                records.append({'path':rel,'size':info.size,'sha256':reader.digest.hexdigest(),'sqlite_online_snapshot':source!=p})
        writer.close()
    except BaseException:
        writer.close()
        write_json(folder/'FAILED.json',{'status':'not_uploadable','reason':'Snapshot preparation interrupted; no manifest committed.'})
        raise
    finally:
        for p in scratch.glob('*.sqlite'):p.unlink()
        scratch.rmdir()
    manifest={'format':'msc-targz-parts-v1','snapshot':stamp,'created_utc':datetime.now(timezone.utc).isoformat(),
      'kind':'incremental' if state.get('last') else 'full','parent':state.get('last'),
      'parts':writer.parts,'files':records,'deleted_since_parent':sorted(set(cache)-set(inventory)),
      'deferred':deferred,'excluded':'Live WAL/SHM, locks, logs, temporary files and backup staging. SQLite copied via online API.',
      'restore':'Download parts in listed order; verify SHA256; concatenate and extract into a fresh directory. Apply parent snapshots first. Deletions are recorded but never automatically applied.'}
    path=folder/f'msc-{stamp}.manifest.json';write_json(path,manifest)
    write_json(folder/'inventory.local.json',inventory)
    state['pending']=str(path.relative_to(ROOT));write_json(STATE,state)
    return plan()

def plan(limit=8):
    state=load_state()
    if not state.get('pending'):return {'status':'idle','last':state.get('last')}
    path=ROOT/state['pending'];m=json.loads(path.read_text(encoding='utf-8'))
    pending=[p for p in m['parts'] if not p.get('drive_id')]
    items=[{'file_uri':str((path.parent/p['name']).resolve()),'file_name':p['name'],'mime_type':'application/octet-stream',
            'parent_folder_id':state['folder_id'],'size':p['size'],'sha256':p['sha256']} for p in pending[:limit]]
    if not pending:items=[{'file_uri':str(path.resolve()),'file_name':path.name,'mime_type':'application/json','parent_folder_id':state['folder_id']}]
    return {'status':'upload_parts' if pending else 'upload_manifest','snapshot':m['snapshot'],'parts_total':len(m['parts']),
            'parts_pending':len(pending),'bytes_total':sum(p['size'] for p in m['parts']),'files':len(m['files']),'uploads':items}

def ack(name,drive_id):
    if not re_id(drive_id):raise ValueError('Invalid Drive file ID')
    state=load_state();path=ROOT/state['pending'];m=json.loads(path.read_text(encoding='utf-8'))
    if name==path.name:
        if any(not p.get('drive_id') for p in m['parts']):raise ValueError('Archive parts must be uploaded before the manifest')
        state['last']={'snapshot':m['snapshot'],'manifest_drive_id':drive_id,'file_name':name}
        state['files']=json.loads((path.parent/'inventory.local.json').read_text(encoding='utf-8'))
        state['pending']=None;write_json(STATE,state)
        write_json(path.parent/'COMPLETE.json',state['last'])
    else:
        found=[p for p in m['parts'] if p['name']==name]
        if len(found)!=1:raise ValueError('Unknown part')
        found[0]['drive_id']=drive_id;write_json(path,m)
    return plan()

def re_id(value):return bool(value) and all(c.isalnum() or c in '_-' for c in value)

def restore(manifest_path,parts_dir,destination):
    m=json.loads(Path(manifest_path).read_text(encoding='utf-8'));destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError('Restore destination must be empty; replay incrementals explicitly into separate staging directories')
    joined=destination.parent/(destination.name+'.archive.tmp')
    try:
        with joined.open('wb') as output:
            for entry in m['parts']:
                p=Path(parts_dir)/entry['name'];h=hashlib.sha256()
                with p.open('rb') as source:
                    while block:=source.read(1024**2):h.update(block);output.write(block)
                if h.hexdigest()!=entry['sha256']:raise ValueError('Archive part checksum mismatch')
        with tarfile.open(joined,'r:gz') as archive:
            for member in archive:
                target=(destination/member.name).resolve()
                if not target.is_relative_to(destination) or not member.isfile():raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        for entry in m['files']:
            h=hashlib.sha256()
            with (destination/entry['path']).open('rb') as f:
                while block:=f.read(1024**2):h.update(block)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Restored file checksum mismatch')
    finally:
        if joined.exists():joined.unlink()
    return {'status':'restored','files':len(m['files']),'snapshot':m['snapshot']}

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--part-mib',type=int,default=32)
    a=sub.add_parser('plan');a.add_argument('--limit',type=int,default=8)
    a=sub.add_parser('ack');a.add_argument('--name',required=True);a.add_argument('--drive-id',required=True)
    a=sub.add_parser('restore');a.add_argument('--manifest',required=True);a.add_argument('--parts-dir',required=True);a.add_argument('--destination',required=True)
    a=p.parse_args()
    with exclusive_run():
        if a.command=='prepare':result=prepare(a.part_mib)
        elif a.command=='plan':result=plan(a.limit)
        elif a.command=='ack':result=ack(a.name,a.drive_id)
        else:result=restore(a.manifest,a.parts_dir,a.destination)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
