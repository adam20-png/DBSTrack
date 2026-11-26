"""
Tune the tracker hyper-parameters of a checkpoint on the HELD-OUT VALIDATION split only
(models/<run>/val_split.json written by train.py: GOT-10k val + 20 LaSOT Training sequences never trained on).
The test benchmarks (OTB100, LaSOT test, UAV123, ...) are never used here.

    python tune.py                   (edit the CONFIGURATION block first)

Two stages of coordinate search (one pass each by default, as for the reference run):
  base : window_influence, penalty_k, test_lr                     -> models/<run>/tune_e<N>/best_params.json
  online: re-detection (lost_thr, lost_instance_sz) and dynamic template (upd_w, upd_thr, upd_interval),
         on top of the base result; thresholds are taken from the score percentiles of THIS model
                                                                    -> models/<run>/tune_e<N>/best_params_online.json
Metric: mean success AUC (21 IoU thresholds) over the validation sequences.
Every trial is logged in models/<run>/tune_e<N>/tune_log_<stage>.json.

Then set PARAMS_FILE in test.py to models/<run>/tune_e<N>/best_params_online.json (or copy its values into TRACKER_PARAMS).
"""
from __future__ import absolute_import, print_function

import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from got10k.datasets import GOT10k                                    # noqa: E402
from tracker.lasot import LaSOTTrain                                    # noqa: E402
from tracker.model import net_from_state_dict                        # noqa: E402
from tracker.split import filter_lasot, load_split                      # noqa: E402
from tracker.tracker import Tracker                                # noqa: E402

# ==================================================================== CONFIGURATION — edit, then run
RUN_NAME = 'dbstrack_ft'  # folder models/<RUN_NAME>/ written by train.py
EPOCH = 20                       # tunes models/<RUN_NAME>/model_e<EPOCH>.pth
CHECKPOINT = ''                  # optional: any other .pth file (then RUN_NAME / EPOCH are ignored)
SPLIT_FILE = ''                 # '' = val_split.json next to the checkpoint
DATA_LASOT = os.path.expanduser('~/data/LaSOT')
STAGES = ('base', 'online')         # run both, or only one of them
PASSES = 1                      # coordinate-search passes per stage
MAX_LASOT_FRAMES = {'base': 1500, 'online': 3000}   # long LaSOT validation sequences are truncated (speed)
INSTANCE_SZ = 303

GRID_BASE = {'window_influence': [0.15, 0.21, 0.30, 0.40, 0.50],
             'penalty_k': [0.02, 0.04, 0.08, 0.15, 0.25],
             'test_lr': [0.30, 0.45, 0.60, 0.80]}
START_BASE = {'window_influence': 0.21, 'penalty_k': 0.04, 'test_lr': 0.52}
# lost_thr / upd_thr candidates are replaced by score percentiles of the model (see main)
GRID_ONLINE = {'lost_thr': None,
           'lost_instance_sz': [399, 463, 527],
           'upd_w': [0.0, 0.15, 0.30, 0.50],
           'upd_thr': None,
           'upd_interval': [5, 10, 25]}
START_ONLINE = {'lost_thr': 0.0, 'lost_instance_sz': 463, 'upd_w': 0.0, 'upd_thr': 0.6, 'upd_interval': 10}
# ====================================================================


def iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def seq_auc(boxes, anno):
    valid = (anno[:, 2] > 0) & (anno[:, 3] > 0)          # frames where the target is annotated
    o = iou(boxes[valid], anno[valid])
    return float(np.mean([(o > t).mean() for t in np.linspace(0, 1, 21)]))


def load_validation(split, max_lasot_frames):
    got = GOT10k(split['got10k_val'], subset='val', return_meta=False)
    las = filter_lasot(LaSOTTrain(DATA_LASOT, subset='Training', return_meta=False), split['lasot'], keep=True)
    seqs = []
    for ds in (got, las):
        for i in range(len(ds)):
            f, a = ds[i][:2]
            a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
            n = min(len(f), len(a), max_lasot_frames)
            seqs.append((ds.seq_names[i], f[:n], a[:n]))
    print('validation: %d GOT-10k val + %d LaSOT sequences, %d frames'
          % (len(got), len(las), sum(len(s[1]) for s in seqs)))
    return seqs


def run_stage(stage, net, dev, seqs, ck_dir):
    cache, log, all_scores = {}, [], []

    def evaluate(p):
        key = tuple(sorted(p.items()))
        if key in cache:
            return cache[key]
        tr = Tracker(net=net, device=dev, instance_sz=INSTANCE_SZ, **p)
        t0 = time.time()
        aucs = []
        all_scores.clear()
        for _, f, a in seqs:
            boxes, _ = tr.track(f, a[0])
            aucs.append(seq_auc(boxes, a))
            all_scores.extend(tr.scores)
        score = float(np.mean(aucs))
        cache[key] = score
        log.append({'params': p, 'auc': score, 'seconds': round(time.time() - t0)})
        print('  %s -> val AUC %.2f  (%.0fs)' % (p, 100 * score, time.time() - t0))
        sys.stdout.flush()
        json.dump(log, open(os.path.join(ck_dir, 'tune_log_%s.json' % stage), 'w'), indent=1)
        return score

    if stage == 'base':
        grid, best, out_name = dict(GRID_BASE), dict(START_BASE), 'best_params.json'
    else:
        bp = json.load(open(os.path.join(ck_dir, 'best_params.json')))
        best = {k: bp[k] for k in START_BASE}
        best.update(START_ONLINE)
        grid, out_name = dict(GRID_ONLINE), 'best_params_online.json'
    best_score = evaluate(best)
    print('[%s] start: %s val AUC %.2f' % (stage, best, 100 * best_score))
    if stage == 'online':
        pc = {q: float(np.percentile(all_scores, q)) for q in (1, 3, 5, 10, 20, 30, 50, 70)}
        print('best-score percentiles on validation:', {k: round(v, 3) for k, v in pc.items()})
        grid['lost_thr'] = [0.0] + [round(pc[q], 4) for q in (1, 3, 5, 10, 20)]
        grid['upd_thr'] = [round(pc[q], 4) for q in (30, 50, 70)]
        best['upd_thr'] = round(pc[50], 4)        # so that upd_w is tuned with an update threshold that fires
        cache.clear()
    for ps in range(PASSES):
        for k, values in grid.items():
            if (k in ('upd_thr', 'upd_interval') and best.get('upd_w', 1) == 0) or \
               (k == 'lost_instance_sz' and best.get('lost_thr', 1) == 0):
                continue                          # no effect while the corresponding mechanism is off
            for v in values:
                p = dict(best)
                p[k] = v
                s = evaluate(p)
                if s > best_score + 1e-6:
                    best, best_score = p, s
            print('[%s] pass %d, after %s: best %s val AUC %.2f' % (stage, ps + 1, k, best, 100 * best_score))
    out = dict(best)
    out['val_auc'] = best_score
    out['instance_sz'] = INSTANCE_SZ
    json.dump(out, open(os.path.join(ck_dir, out_name), 'w'), indent=1)
    print('[%s] BEST %s -> %s' % (stage, out, os.path.join(ck_dir, out_name)))


def main():
    ck = os.path.abspath(CHECKPOINT) if CHECKPOINT else os.path.join(ROOT, 'models', RUN_NAME, 'model_e%d.pth' % EPOCH)
    if not os.path.isfile(ck):
        sys.exit('checkpoint not found: ' + ck)
    # one output folder per epoch: models/<run>/tune_e<N>/ (the split is read from models/<run>/val_split.json)
    stem = os.path.splitext(os.path.basename(ck))[0]
    out_dir = os.path.join(os.path.dirname(ck), 'tune_' + stem.split('_')[-1])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    split = load_split(SPLIT_FILE or os.path.join(os.path.dirname(ck), 'val_split.json'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd = torch.load(ck, map_location='cpu')
    net = net_from_state_dict(sd.get('net', sd))
    net.to(dev).eval()
    print('checkpoint:', ck, '| results ->', out_dir)
    for stage in STAGES:
        run_stage(stage, net, dev, load_validation(split, MAX_LASOT_FRAMES[stage]), out_dir)


if __name__ == '__main__':
    main()
