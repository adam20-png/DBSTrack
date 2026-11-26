"""
LaSOT evaluation following pytracking's protocol (pytracking/analysis/extract_results.py + plot_results.py), which
is the Python reference used by most recent tracking papers and reproduces the official LaSOT toolkit.

Differences with the got10k-style evaluation of benchmarks.py (OTB machinery):
  * frames where the target is fully occluded or out of view (full_occlusion.txt / out_of_view.txt) are COUNTED
    as failures (IoU = -1, centre error = inf), instead of being scored against the box;
  * pixel convention "+1" for IoU and box centres ((w - 1) / 2), as in the MATLAB toolkit;
  * predicted boxes with zero width/height are replaced by the previous box.
Scores: AUC = mean success over IoU thresholds 0:0.05:1, Prec = precision at 20 px,
NormPrec = normalized precision at 0.2 (value of the curve, as printed by pytracking).

Usage:  python -m tracker.lasot_official <lasot_root> <result_dir> [<result_dir> ...]
        (<result_dir> contains one <sequence>.txt per test sequence, boxes x,y,w,h)
"""
from __future__ import absolute_import, print_function

import os
import sys

import numpy as np

LASOT_TEST = """airplane-1 airplane-9 airplane-13 airplane-15 basketball-1 basketball-6 basketball-7 basketball-11
bear-2 bear-4 bear-6 bear-17 bicycle-2 bicycle-7 bicycle-9 bicycle-18 bird-2 bird-3 bird-15 bird-17 boat-3 boat-4
boat-12 boat-17 book-3 book-10 book-11 book-19 bottle-1 bottle-12 bottle-14 bottle-18 bus-2 bus-5 bus-17 bus-19
car-2 car-6 car-9 car-17 cat-1 cat-3 cat-18 cat-20 cattle-2 cattle-7 cattle-12 cattle-13 spider-14 spider-16
spider-18 spider-20 coin-3 coin-6 coin-7 coin-18 crab-3 crab-6 crab-12 crab-18 surfboard-12 surfboard-4 surfboard-5
surfboard-8 cup-1 cup-4 cup-7 cup-17 deer-4 deer-8 deer-10 deer-14 dog-1 dog-7 dog-15 dog-19 guitar-3 guitar-8
guitar-10 guitar-16 person-1 person-5 person-10 person-12 pig-2 pig-10 pig-13 pig-18 rubicCube-1 rubicCube-6
rubicCube-14 rubicCube-19 swing-10 swing-14 swing-17 swing-20 drone-13 drone-15 drone-2 drone-7 pool-12 pool-15
pool-3 pool-7 rabbit-10 rabbit-13 rabbit-17 rabbit-19 racing-10 racing-15 racing-16 racing-20 robot-1 robot-19
robot-5 robot-8 sepia-13 sepia-16 sepia-6 sepia-8 sheep-3 sheep-5 sheep-7 sheep-9 skateboard-16 skateboard-19
skateboard-3 skateboard-8 tank-14 tank-16 tank-6 tank-9 tiger-12 tiger-18 tiger-4 tiger-6 train-1 train-11 train-20
train-7 truck-16 truck-3 truck-6 truck-7 turtle-16 turtle-5 turtle-8 turtle-9 umbrella-17 umbrella-19 umbrella-2
umbrella-9 yoyo-15 yoyo-17 yoyo-19 yoyo-7 zebra-10 zebra-14 zebra-16 zebra-17 elephant-1 elephant-12 elephant-16
elephant-18 goldfish-3 goldfish-7 goldfish-8 goldfish-10 hat-1 hat-2 hat-5 hat-18 kite-4 kite-6 kite-10 kite-15
motorcycle-1 motorcycle-3 motorcycle-9 motorcycle-18 mouse-1 mouse-8 mouse-9 mouse-17 flag-3 flag-9 flag-5 flag-2
frog-3 frog-4 frog-20 frog-9 gametarget-1 gametarget-2 gametarget-7 gametarget-13 hand-2 hand-3 hand-9 hand-16
helmet-5 helmet-11 helmet-19 helmet-13 licenseplate-6 licenseplate-12 licenseplate-13 licenseplate-15 electricfan-1
electricfan-10 electricfan-18 electricfan-20 chameleon-3 chameleon-6 chameleon-11 chameleon-20 crocodile-3
crocodile-4 crocodile-10 crocodile-14 gecko-1 gecko-5 gecko-16 gecko-19 fox-2 fox-3 fox-5 fox-20 giraffe-2
giraffe-10 giraffe-13 giraffe-15 gorilla-4 gorilla-6 gorilla-9 gorilla-13 hippo-1 hippo-7 hippo-9 hippo-20 horse-1
horse-4 horse-12 horse-15 kangaroo-2 kangaroo-5 kangaroo-11 kangaroo-14 leopard-1 leopard-7 leopard-16 leopard-20
lion-1 lion-5 lion-12 lion-20 lizard-1 lizard-3 lizard-6 lizard-13 microphone-2 microphone-6 microphone-14
microphone-16 monkey-3 monkey-4 monkey-9 monkey-17 shark-2 shark-3 shark-5 shark-6 squirrel-8 squirrel-11
squirrel-13 squirrel-19 volleyball-1 volleyball-13 volleyball-18 volleyball-19""".split()

THR_IOU = np.arange(0.0, 1.0 + 0.05, 0.05)
THR_CENTER = np.arange(0, 51, dtype=np.float64)
THR_NORM = np.arange(0, 51, dtype=np.float64) / 100.0


def _load(path, delimiters=(',', '\t', ' ')):
    for d in delimiters:
        try:
            return np.loadtxt(path, delimiter=d, dtype=np.float64)
        except ValueError:
            pass
    raise ValueError('cannot read ' + path)


def _center(bb):
    return bb[:, :2] + 0.5 * (bb[:, 2:] - 1.0)


def seq_curves(pred, anno, visible):
    pred = pred.copy().reshape(-1, 4)
    anno = anno.reshape(-1, 4)
    if np.isnan(pred).any() or (pred[:, 2:] < 0).any():
        raise ValueError('invalid predicted boxes')
    for i in range(1, len(pred)):                                  # zero-size box -> previous box
        if (pred[i, 2:] == 0).any() and not np.isnan(anno[i]).any():
            pred[i] = pred[i - 1]
    if len(pred) > len(anno):
        pred = pred[:len(anno)]
    elif len(pred) < len(anno):
        raise ValueError('fewer predictions than ground-truth frames')
    pred[0] = anno[0]
    valid = ((anno > 0).sum(1) == 4) & visible
    tl = np.maximum(pred[:, :2], anno[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, anno[:, :2] + anno[:, 2:] - 1.0)
    inter = np.clip(br - tl + 1.0, 0, None).prod(1)
    union = pred[:, 2:].prod(1) + anno[:, 2:].prod(1) - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / union
        err_c = np.sqrt(((_center(pred) - _center(anno)) ** 2).sum(1))
        err_n = np.sqrt(((_center(pred) / anno[:, 2:] - _center(anno) / anno[:, 2:]) ** 2).sum(1))
    iou[~valid] = -1.0
    err_c[~valid] = np.inf
    err_n[~valid] = -1.0
    err_n[~visible] = np.inf
    err_c[~visible] = np.inf
    n = len(anno)                                                  # exclude_invalid_frames=False (default)
    return ((iou[:, None] > THR_IOU[None]).sum(0) / n,
            (err_c[:, None] <= THR_CENTER[None]).sum(0) / n,
            (err_n[:, None] <= THR_NORM[None]).sum(0) / n)


def evaluate(lasot_root, result_dir):
    S, P, N = [], [], []
    for name in LASOT_TEST:
        d = os.path.join(lasot_root, name.split('-')[0], name)
        if not os.path.isdir(d):
            d = os.path.join(lasot_root, 'Testing', name.split('-')[0], name)
        anno = _load(os.path.join(d, 'groundtruth.txt'))
        occ = _load(os.path.join(d, 'full_occlusion.txt')).reshape(-1)
        oov = _load(os.path.join(d, 'out_of_view.txt')).reshape(-1)
        visible = (occ == 0) & (oov == 0)
        pred = _load(os.path.join(result_dir, name + '.txt'))
        s, p, n = seq_curves(pred, anno, visible)
        S.append(s), P.append(p), N.append(n)
    S, P, N = [100 * np.mean(x, 0) for x in (S, P, N)]
    return {'AUC': round(float(S.mean()), 2), 'OP50': round(float(S[10]), 2), 'OP75': round(float(S[15]), 2),
            'Prec': round(float(P[20]), 2), 'NormPrec': round(float(N[20]), 2)}


if __name__ == '__main__':
    root = sys.argv[1]
    print('%-90s %6s %6s %6s %6s %8s' % ('results (pytracking LaSOT protocol)', 'AUC', 'OP50', 'OP75', 'Prec', 'NormPrec'))
    for rd in sys.argv[2:]:
        try:
            r = evaluate(root, rd)
            print('%-90s %6.2f %6.2f %6.2f %6.2f %8.2f' % (rd[-90:], r['AUC'], r['OP50'], r['OP75'], r['Prec'],
                                                           r['NormPrec']))
        except Exception as e:
            print('%-90s ERROR %s' % (rd[-90:], e))
