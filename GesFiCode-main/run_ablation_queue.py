#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融全包调度器:GPU 队列 + 固定并发,自动跑完一组 run。
用法: python run_ablation_queue.py <jobset>
  jobset: xrf | widar
每张可用 GPU 同时跑 PER_GPU 个任务。结果在 log/<exp_id>/bestacc.txt。
"""
import os, sys, time, subprocess, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
XRF = '/mnt/nj-1/usr/liujia7/workspace/datasets/xrf55_processed_data'
WIDAR1AP = '/mnt/nj-1/usr/liujia7/workspace/datasets/widar_processed_data/Processed_Widar_1AP_Batch'
LOGDIR = os.path.join(HERE, 'ablation_logs')
os.makedirs(LOGDIR, exist_ok=True)

GPUS = [0, 1, 2, 3]      # 允许使用的 GPU
PER_GPU = 3              # 每卡并发数
ABLATIONS = ['M0', 'M1', 'M2', 'M4', 'full']   # M4 = w/o方向解耦(论文M3)
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCH = 50


def build_jobs(jobset):
    jobs = []
    if jobset == 'xrf':
        for exp in ['cross_user', 'cross_env']:
            for abl in ABLATIONS:
                for sd in SEEDS:
                    eid = f'abl_xrf_{exp}_{abl}_s{sd}'
                    extra = [] if exp != 'cross_env' else ['--test_scene', '4']
                    cmd = ['python', '-u', 'csimain.py', '--dataset', 'xrf55',
                           '--experiment', exp, '--ablation', abl, '--seed', str(sd),
                           '--data_path', XRF, '--max_epoch', str(MAX_EPOCH),
                           '--exp_id', eid] + extra
                    jobs.append((eid, cmd))
    elif jobset in ('widar', 'widar2'):
        # Widar 1AP rx1, 固定划分(代码默认即 L1-4/L5, O1-4/O5, E1+2/E3, U05-15/U16-17)
        exps = ['cross_loc', 'cross_user'] if jobset == 'widar2' \
            else ['cross_loc', 'cross_ori', 'cross_env', 'cross_user']
        for exp in exps:
            for abl in ABLATIONS:
                for sd in SEEDS:
                    eid = f'abl_widar1ap_{exp}_{abl}_s{sd}'
                    cmd = ['python', '-u', 'csimain.py', '--dataset', 'widar',
                           '--experiment', exp, '--ablation', abl, '--seed', str(sd),
                           '--rx', 'RX1', '--data_path', WIDAR1AP,
                           '--max_epoch', str(MAX_EPOCH), '--exp_id', eid]
                    jobs.append((eid, cmd))
    else:
        raise ValueError(jobset)
    return jobs


def done(eid):
    return os.path.exists(os.path.join(HERE, 'log', eid, 'bestacc.txt'))


def main():
    jobset = sys.argv[1]
    jobs = build_jobs(jobset)
    # 跳过已完成
    pending = [(e, c) for e, c in jobs if not done(e)]
    print(f'[Queue] jobset={jobset}  total={len(jobs)}  pending={len(pending)}  '
          f'(skip {len(jobs)-len(pending)} done)', flush=True)

    slots = [(g, i) for g in GPUS for i in range(PER_GPU)]  # 12 slots
    running = {}   # slot -> (eid, proc)
    it = iter(pending)
    finished = 0
    total = len(pending)

    while True:
        # 填充空 slot
        for slot in slots:
            if slot in running:
                continue
            try:
                eid, cmd = next(it)
            except StopIteration:
                break
            gpu = slot[0]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            logf = open(os.path.join(LOGDIR, eid + '.log'), 'w')
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[slot] = (eid, p, logf)
            print(f'[start] GPU{gpu} slot{slot[1]}  {eid}', flush=True)
            time.sleep(2)
        if not running:
            break
        # 轮询完成
        time.sleep(20)
        for slot in list(running.keys()):
            eid, p, logf = running[slot]
            if p.poll() is not None:
                logf.close()
                ok = done(eid)
                finished += 1
                print(f'[done {finished}/{total}] {eid}  '
                      f'{"OK bestacc=" + open(os.path.join(HERE,"log",eid,"bestacc.txt")).read().strip() if ok else "FAILED"}',
                      flush=True)
                del running[slot]

    print(f'[Queue] ALL DONE jobset={jobset}', flush=True)


if __name__ == '__main__':
    main()
