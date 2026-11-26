"""
Evaluate a DBSTrack checkpoint on ONE test benchmark per run (one-pass evaluation, got10k toolkit).

    python test.py                   (set BENCHMARK in the CONFIGURATION block, run, then change it and run again)

Benchmarks (choose one in BENCHMARK):
  OTB100, LaSOT (280 test sequences), UAV123, DTB70, TC128 (Temple-Color-128), NFS30 (Need for Speed, 30 fps)
      -> AUC (success, 21 IoU thresholds), Prec (precision at 20 px), FPS; LaSOT also NormPrec (at 0.2)
  GOT10k-test -> no local ground truth: a zip is written, submit it to http://got-10k.aitestunion.com

Checkpoint: models/dbstrack.pth (published model), or models/<RUN_NAME>/model_e<EPOCH>.pth, or any .pth file.

Results, grouped by BENCHMARK, one folder per tested epoch, named <MODEL_NAME>_e<EPOCH>[_<LABEL>]:
  results/<BENCHMARK>/dbstrack_e80/summary.json        scores + checkpoint + tracker parameters
  results/<BENCHMARK>/dbstrack_e80/results/...         raw boxes (x y w h) per sequence
  results/<BENCHMARK>/dbstrack_e80/reports/...         success / precision plots (tracker name: MODEL_NAME)
  results/all_results.csv                            ONE LINE PER TEST
got10k skips sequences whose result file already exists (an interrupted run resumes where it stopped). So if the
folder dbstrack_e80 already holds results of ANOTHER checkpoint or of OTHER tracker parameters, the new test goes to
dbstrack_e80_2 (then _3, ...) instead of silently reusing them.
"""
from __future__ import absolute_import, print_function

import csv
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tracker.benchmarks import run_benchmark          # noqa: E402
from tracker.tracker import Tracker              # noqa: E402

# ==================================================================== CONFIGURATION — edit, then run
RUN_NAME = 'dbstrack_ft'         # used only when CHECKPOINT is ''
EPOCH = 20                       # tests models/<RUN_NAME>/model_e<EPOCH>.pth when CHECKPOINT is ''
CHECKPOINT = os.path.join(ROOT, 'models', 'dbstrack.pth')    # published model ('' = use RUN_NAME / EPOCH)
LABEL = ''                       # optional free text added to the result folder name, e.g. 'noupdate'
MODEL_NAME = 'DBSTrack'         # name of the result folders (<MODEL_NAME>_e<EPOCH>) and of the tracker in the plots

# Tracker parameters of the reference model (chosen on the validation split, see tune.py).
# Edit the values directly, or set PARAMS_FILE to a JSON written by tune.py (it then replaces this dict).
TRACKER_PARAMS = {
    'instance_sz': 303,          # search region (px); score map 23 x 23
    'window_influence': 0.15,    # weight of the cosine window
    'penalty_k': 0.04,           # scale / aspect-ratio change penalty
    'test_lr': 0.8,              # box size update rate
    'lost_thr': 0.0245,          # re-detection: target considered lost below this score (0 = off)
    'lost_instance_sz': 527,     # re-detection: enlarged search region while lost
    'upd_w': 0.5,                # dynamic template weight (0 = off, first-frame template only)
    'upd_thr': 0.4602,           # dynamic template: minimum score to update
    'upd_interval': 10,          # dynamic template: update every N frames
}
PARAMS_FILE = ''                 # e.g. os.path.join(ROOT, 'models', 'dbstrack_ft', 'tune_e20', 'best_params_online.json')

BENCHMARK = 'OTB100'        # ONE of: 'OTB100', 'LaSOT', 'UAV123', 'DTB70', 'TC128', 'NFS30', 'TrackingNet', 'GOT10k-test'
DATA_ROOTS = {
    'OTB100': os.path.expanduser('~/data/OTB100'),           # searched automatically if wrong
    'LaSOT': os.path.expanduser('~/data/LaSOT'),                              # contains Testing/
    'UAV123': os.path.expanduser('~/data/UAV123'),           # contains anno/ and data_seq/
    'DTB70': os.path.expanduser('~/data/DTB70'),
    'TC128': os.path.expanduser('~/data/Temple-color-128'),
    'NFS30': os.path.expanduser('~/data/NFS'),                            # <seq>/30/<seq>/*.jpg + <seq>/30/<seq>.txt
    'TrackingNet': os.path.expanduser('~/data/TrackingNet'),                  # contains TEST/anno and TEST/frames
    'GOT10k-test': os.path.expanduser('~/data/GOT-10k'),     # contains test/
}
VISUALIZE = False
OUT_DIR = os.path.join(ROOT, 'results')
# ====================================================================

PARAM_KEYS = ('instance_sz', 'window_influence', 'penalty_k', 'test_lr', 'lost_thr', 'lost_instance_sz',
              'upd_w', 'upd_thr', 'upd_interval')


def resolve_checkpoint():
    """-> (checkpoint path, epoch label)"""
    if CHECKPOINT:
        ck = os.path.abspath(os.path.expanduser(CHECKPOINT))
        stem = os.path.splitext(os.path.basename(ck))[0]
        m = re.search(r'_e(\d+)$', stem)
        return ck, 'e%s' % m.group(1) if m else stem
    ck = os.path.join(ROOT, 'models', RUN_NAME, 'model_e%d.pth' % EPOCH)
    if not os.path.isfile(ck):
        avail = sorted(int(re.search(r'_e(\d+)\.pth$', f).group(1))
                       for f in glob.glob(os.path.join(ROOT, 'models', RUN_NAME, 'model_e*.pth')))
        sys.exit('checkpoint not found: %s\navailable epochs for %s: %s' % (ck, RUN_NAME, avail or 'none'))
    return ck, 'e%d' % EPOCH


def result_folder(bench, base, ck, params):
    """results/<bench>/<base>, or <base>_2, _3 ... if <base> holds another checkpoint / other parameters."""
    k = 1
    while True:
        d = os.path.join(OUT_DIR, bench, base if k == 1 else '%s_%d' % (base, k))
        f = os.path.join(d, 'summary.json')
        if not os.path.isfile(f):
            return d
        old = json.load(open(f))
        if old.get('checkpoint') == ck and old.get('tracker_params') == params:
            return d
        k += 1


def append_csv(row):
    path = os.path.join(OUT_DIR, 'all_results.csv')
    cols = ['date', 'model', 'epoch', 'label', 'benchmark', 'AUC', 'Prec', 'NormPrec', 'FPS', 'folder',
            'checkpoint', 'params']
    new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in cols})
    return path


def main():
    if BENCHMARK not in DATA_ROOTS:
        sys.exit('unknown BENCHMARK %r, choose one of %s' % (BENCHMARK, list(DATA_ROOTS)))
    params = dict(TRACKER_PARAMS)
    if PARAMS_FILE:
        params = {k: v for k, v in json.load(open(PARAMS_FILE)).items() if k in PARAM_KEYS}
    ck, ep = resolve_checkpoint()
    base = '%s_%s%s' % (MODEL_NAME, ep, '_' + LABEL if LABEL else '')
    run_dir = result_folder(BENCHMARK, base, ck, params)
    name = os.path.basename(run_dir)
    res_dir, rep_dir = os.path.join(run_dir, 'results'), os.path.join(run_dir, 'reports')
    os.makedirs(run_dir, exist_ok=True)
    summary_file = os.path.join(run_dir, 'summary.json')
    info = {'benchmark': BENCHMARK, 'model': MODEL_NAME, 'epoch': ep, 'label': LABEL, 'checkpoint': ck,
            'tracker_params': params}
    if not os.path.isfile(summary_file):                 # written first, so that an interrupted run resumes here
        json.dump(info, open(summary_file, 'w'), indent=2)
    print('=' * 78)
    print('TEST  %s  |  %s  (%s, epoch %s)' % (BENCHMARK, name, MODEL_NAME, ep[1:] if ep[:1] == 'e' else ep))
    print('checkpoint :', ck)
    print('parameters :', params)
    print('results    :', run_dir)
    print('=' * 78)

    tracker = Tracker(net_path=ck, name=MODEL_NAME, **params)
    root = DATA_ROOTS[BENCHMARK]
    res = run_benchmark(BENCHMARK, root, tracker, res_dir, rep_dir, visualize=VISUALIZE)
    res['date'] = time.strftime('%Y-%m-%d %H:%M')
    json.dump(dict(info, scores=res), open(summary_file, 'w'), indent=2)
    csv_file = append_csv(dict(res, model=MODEL_NAME, epoch=ep, label=LABEL, benchmark=BENCHMARK,
                               folder=os.path.relpath(run_dir, OUT_DIR), checkpoint=ck, params=json.dumps(params),
                               AUC=res.get('AUC', res.get('submission', ''))))

    # every benchmark already evaluated with this checkpoint + these parameters
    print('\n%s  (%s)' % (name, ck))
    print('%-12s %6s %6s %9s %7s' % ('benchmark', 'AUC', 'Prec', 'NormPrec', 'FPS'))
    for bench in DATA_ROOTS:
        s = {}
        for f in glob.glob(os.path.join(OUT_DIR, bench, '*', 'summary.json')):
            d = json.load(open(f))
            if d.get('checkpoint') == ck and d.get('tracker_params') == params and d.get('label', '') == LABEL:
                s = d.get('scores', {})
        if 'AUC' in s:
            print('%-12s %6.1f %6.1f %9s %7.1f' % (bench, s['AUC'], s['Prec'],
                                                   '%.1f' % s['NormPrec'] if 'NormPrec' in s else '-', s['FPS']))
        elif 'submission' in s:
            print('%-12s submit %s' % (bench, s['submission']))
    print('saved to', summary_file)
    print('all tests :', csv_file)


if __name__ == '__main__':
    main()
