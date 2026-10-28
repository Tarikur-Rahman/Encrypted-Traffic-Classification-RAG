"""SHA-256 snapshot of all project files except this experiment directory."""
import csv,hashlib,json,os,sys
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def snapshot(label):
    count=0
    with (OUT/f'audit/project_hashes_{label}.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['path','size_bytes','sha256'])
        for directory,dirs,files in os.walk(ROOT):
            dirs[:]=sorted(d for d in dirs if Path(directory)/d!=OUT)
            for name in sorted(files):
                p=Path(directory)/name;w.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,sha(p)]);count+=1
                if count%25000==0:print(label,count,flush=True)
    print(label,'COMPLETE',count,flush=True)
def compare():
    def read(label):
        with (OUT/f'audit/project_hashes_{label}.csv').open(encoding='utf-8',newline='') as f:return {r['path']:r for r in csv.DictReader(f)}
    before,after=read('before'),read('after')
    changed=[p for p in before if before[p]!=after.get(p)];added=sorted(set(after)-set(before))
    work=[p for p in changed+added if not p.startswith('.git/')]
    report=f'# Main-project change report\n\nExisting main-project working files modified/deleted: {sum(not p.startswith(".git/") for p in changed)}\n\nWorking files added outside Table 3: {sum(not p.startswith(".git/") for p in added)}\n\n{len(before)} before / {len(after)} after files hashed, excluding only table3_retrieval_quality. Table 1 and Table 2 included without exemptions.\n\nGit metadata reported separately. Full differences:\n\n```json\n'+json.dumps(dict(changed=changed,added=added),indent=2)+'\n```\n'
    (OUT/'audit/main_project_change_report.md').write_text(report,encoding='utf-8')
    assert not work,report
    print(report)
if __name__=='__main__':
    if sys.argv[1]=='compare':compare()
    else:snapshot(sys.argv[1])
