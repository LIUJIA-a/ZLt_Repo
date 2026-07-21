#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主实验调度器: Widar 6AP LODO + XRF55 LODO, 3种子。
用法: python run_main_exp.py
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W6AP = '/mnt/nj-1/usr/liujia7/workspace/datasets/widar_processed_data/Processed_Widar_6AP'
XRF  = '/mnt/nj-1/usr/liujia7/workspace/datasets/xrf55_processed_data'

GPUS    = [2, 3]
PER_GPU = 3
SEEDS   = [0, 1, 2]
MAX_EPOCH = 50


def build_jobs():
    jobs = []

    # ── Widar 6AP ──────────────────────────────────────────────────────────
    # cross_loc: 留一法 L1-L5, 5折
    for test_l in ['L1','L2','L3','L4','L5']:
        train_ls = ','.join(l for l in ['L1','L2','L3','L4','L5'] if l != test_l)
        for sd in SEEDS:
            eid = f'main_widar6ap_cloc_test{test_l}_s{sd}'
            cmd = ['python','-u','csimain.py','--dataset','widar',
                   '--experiment','cross_loc','--ablation','full',
                   '--seed',str(sd),'--train_envs',train_ls,
                   '--data_path',W6AP,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
            jobs.append((eid, cmd))

    # cross_ori: 留一法 O1-O5, 5折
    for test_o in ['O1','O2','O3','O4','O5']:
        train_os = ','.join(o for o in ['O1','O2','O3','O4','O5'] if o != test_o)
        for sd in SEEDS:
            eid = f'main_widar6ap_cori_test{test_o}_s{sd}'
            cmd = ['python','-u','csimain.py','--dataset','widar',
                   '--experiment','cross_ori','--ablation','full',
                   '--seed',str(sd),'--train_envs',train_os,
                   '--data_path',W6AP,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
            jobs.append((eid, cmd))

    # cross_env: 留一法 E1/E2/E3, 3折
    for test_e in ['E1','E2','E3']:
        train_es = ','.join(e for e in ['E1','E2','E3'] if e != test_e)
        for sd in SEEDS:
            eid = f'main_widar6ap_cenv_test{test_e}_s{sd}'
            cmd = ['python','-u','csimain.py','--dataset','widar',
                   '--experiment','cross_env','--ablation','full',
                   '--seed',str(sd),'--train_envs',train_es,
                   '--data_path',W6AP,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
            jobs.append((eid, cmd))

    # cross_user: 固定划分 U05-U15/U16-U17, 3种子
    for sd in SEEDS:
        eid = f'main_widar6ap_cuser_s{sd}'
        cmd = ['python','-u','csimain.py','--dataset','widar',
               '--experiment','cross_user','--ablation','full',
               '--seed',str(sd),
               '--data_path',W6AP,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
        jobs.append((eid, cmd))

    # ── XRF55 ──────────────────────────────────────────────────────────────
    # cross_user: scene1 固定 U01-24/U25-30
    for sd in SEEDS:
        eid = f'main_xrf_cuser_s{sd}'
        cmd = ['python','-u','csimain.py','--dataset','xrf55',
               '--experiment','cross_user','--ablation','full',
               '--seed',str(sd),
               '--data_path',XRF,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
        jobs.append((eid, cmd))

    # cross_env: scene4测试
    for sd in SEEDS:
        eid = f'main_xrf_cenv_s{sd}'
        cmd = ['python','-u','csimain.py','--dataset','xrf55',
               '--experiment','cross_env','--ablation','full',
               '--seed',str(sd),'--test_scene','4',
               '--data_path',XRF,'--max_epoch',str(MAX_EPOCH),'--exp_id',eid]
        jobs.append((eid, cmd))

    return jobs


def done(eid):
    return os.path.exists(os.path.join(HERE, 'log', eid, 'bestacc.txt'))


def main():
    jobs = build_jobs()
    pending = [(e, c) for e, c in jobs if not done(e)]
    print(f'[Queue] total={len(jobs)} pending={len(pending)} (skip {len(jobs)-len(pending)} done)', flush=True)

    slots = [(g, i) for g in GPUS for i in range(PER_GPU)]
    running = {}
    it = iter(pending); finished = 0; total = len(pending)
    os.makedirs(os.path.join(HERE, 'main_logs'), exist_ok=True)

    while True:
        for slot in slots:
            if slot in running:
                continue
            try:
                eid, cmd = next(it)
            except StopIteration:
                break
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]))
            logf = open(os.path.join(HERE, 'main_logs', eid+'.log'), 'w')
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[slot] = (eid, p, logf)
            print(f'[start] GPU{slot[0]} {eid}', flush=True)
            time.sleep(2)
        if not running:
            break
        time.sleep(30)
        for slot in list(running.keys()):
            eid, p, logf = running[slot]
            if p.poll() is not None:
                logf.close(); finished += 1
                b = open(os.path.join(HERE,'log',eid,'bestacc.txt')).read().strip() if done(eid) else 'FAIL'
                print(f'[done {finished}/{total}] {eid} {b}', flush=True)
                del running[slot]

    print('[Queue] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
