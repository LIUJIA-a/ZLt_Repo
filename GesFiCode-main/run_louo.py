#!/usr/bin/env python3
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
XRF  = '/mnt/nj-1/usr/liujia7/workspace/datasets/xrf55_processed_data'
GPUS = [2, 3]; PER_GPU = 3; MAX_EPOCH = 50
SEEDS = list(map(int, sys.argv[1:])) if len(sys.argv)>1 else [0]

jobs = []
for u in range(1, 31):
    for sd in SEEDS:
        eid = f'main_xrf_louo_u{u:02d}_s{sd}'
        cmd = ['python','-u','csimain.py','--dataset','xrf55',
               '--experiment','cross_user','--ablation','full',
               '--seed',str(sd),'--xrf_test_users',str(u),
               '--data_path',XRF,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
        jobs.append((eid, cmd))

done_fn = lambda e: os.path.exists(os.path.join(HERE,'log',e,'bestacc.txt'))
pending = [(e,c) for e,c in jobs if not done_fn(e)]
print(f'[LOUO] total={len(jobs)} pending={len(pending)}', flush=True)

slots = [(g,i) for g in GPUS for i in range(PER_GPU)]
running = {}; it = iter(pending); finished=0; total=len(pending)
os.makedirs(os.path.join(HERE,'main_logs'), exist_ok=True)

while True:
    for slot in slots:
        if slot in running:
            continue
        try: eid,cmd = next(it)
        except StopIteration: break
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]))
        logf = open(os.path.join(HERE,'main_logs',eid+'.log'),'w')
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
        running[slot]=(eid,p,logf)
        print(f'[start] GPU{slot[0]} {eid}', flush=True); time.sleep(1)
    if not running: break
    time.sleep(20)
    for slot in list(running.keys()):
        eid,p,logf = running[slot]
        if p.poll() is not None:
            logf.close(); finished+=1
            b = open(os.path.join(HERE,'log',eid,'bestacc.txt')).read().strip() if done_fn(eid) else 'FAIL'
            print(f'[done {finished}/{total}] {eid} {b}', flush=True)
            del running[slot]

print('[LOUO] ALL DONE', flush=True)
