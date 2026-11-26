"""Benchmark wrappers (got10k toolkit conventions, one-pass evaluation).

Metrics: AUC = success score (21 IoU thresholds), Prec = precision at 20 px, NormPrec = normalized precision at 0.2
(LaSOT), FPS = average tracking speed (image reading / display excluded). GOT-10k test produces the files to submit
to the official evaluation server (http://got-10k.aitestunion.com).
TrackingNet test (511 sequences, only the first-frame box is public) produces a zip to submit to the official
evaluation server (https://eval.ai, TrackingNet challenge): one <sequence>.txt per sequence, boxes x,y,w,h.
"""
from __future__ import absolute_import

import glob
import os
import sys

import numpy as np
from got10k.datasets import NfS, TColor128
from got10k.experiments import ExperimentDTB70, ExperimentGOT10k, ExperimentOTB, ExperimentUAV123

from tracker.lasot import LaSOTTrain


class ExperimentLaSOT(ExperimentOTB):
    """OPE on the 280 LaSOT test sequences with the OTB machinery (got10k 0.1.3 has no working LaSOT experiment)."""

    def __init__(self, lasot_root, result_dir, report_dir):
        self.dataset = LaSOTTrain(lasot_root, subset='Testing', return_meta=False)
        self.result_dir = os.path.join(result_dir, 'LaSOT')
        self.report_dir = os.path.join(report_dir, 'LaSOT')
        self.nbins_iou = 21
        self.nbins_ce = 51

    def norm_precision(self, name):
        thr = np.linspace(0, 0.5, 51)
        curves = []
        for s, (_, anno) in enumerate(self.dataset):
            boxes = np.loadtxt(os.path.join(self.result_dir, name, '%s.txt' % self.dataset.seq_names[s]),
                               delimiter=',').reshape(-1, 4)
            boxes[0] = anno[0]
            valid = (anno[:, 2] > 0) & (anno[:, 3] > 0)
            c_p = boxes[:, :2] + boxes[:, 2:] / 2
            c_g = anno[:, :2] + anno[:, 2:] / 2
            wh = np.maximum(anno[:, 2:], 1)
            err = np.linalg.norm((c_p - c_g) / wh, axis=1)[valid]
            curves.append([(err <= t).mean() for t in thr])
        return float(np.mean(curves, axis=0)[20])        # threshold 0.2


class NfS30(NfS):
    """NfS at 30 fps (100 sequences). Differences with got10k.datasets.NfS(fps=30):
      * extra annotation files such as <seq>_pytracking.txt are ignored (got10k would count them as sequences);
      * robust length matching: the official annotation is given at 240 fps -> 1 row out of 8 is kept, then
        image list and annotation are cut to the same length (got10k asserts and crashes on some sequences)."""

    def __init__(self, root_dir):
        self.fps, self.root_dir = 30, root_dir
        self._check_integrity(root_dir)
        files = sorted(glob.glob(os.path.join(root_dir, '*', '30', '*.txt')))
        self.anno_files = [f for f in files
                           if os.path.basename(f)[:-4] == os.path.basename(os.path.dirname(os.path.dirname(f)))]
        self.seq_names = [os.path.basename(f)[:-4] for f in self.anno_files]
        self.seq_dirs = [os.path.join(os.path.dirname(f), n) for f, n in zip(self.anno_files, self.seq_names)]

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.seq_names.index(index)
        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], dtype=str)
        anno = anno[:, 1:5].astype(float)                 # [left, top, right, bottom]
        anno[:, 2:] -= anno[:, :2]                        # [left, top, width, height]
        if len(anno) != len(img_files) and abs(len(anno) / max(len(img_files), 1) - 8) < 1:
            anno = anno[0::8, :]
        n = min(len(img_files), len(anno))
        if abs(len(img_files) - len(anno)) > 1:
            print('  [NFS30] %s: %d images / %d boxes -> first %d frames used'
                  % (self.seq_names[index], len(img_files), len(anno), n))
        return img_files[:n], anno[:n]


class ExperimentNfS30(ExperimentOTB):
    def __init__(self, root_dir, result_dir, report_dir):
        self.dataset = NfS30(root_dir)
        self.result_dir = os.path.join(result_dir, 'NfS30')
        self.report_dir = os.path.join(report_dir, 'NfS30')
        self.nbins_iou = 21
        self.nbins_ce = 51


class ExperimentTC128(ExperimentOTB):
    """Temple-Color-128 without the automatic download of got10k (the folder must contain the 128 sequences)."""

    def __init__(self, root_dir, result_dir, report_dir):
        self.dataset = TColor128(root_dir, download=False)
        self.result_dir = os.path.join(result_dir, 'TColor128')
        self.report_dir = os.path.join(report_dir, 'TColor128')
        self.nbins_iou = 21
        self.nbins_ce = 51


class TrackingNetTest(object):
    """TrackingNet test set: <root>/TEST/anno/<seq>.txt holds ONLY the first-frame box and
    <root>/TEST/frames/<seq>/<i>.jpg the frames 0..N-1 (numeric order). got10k 0.1.3's TrackingNet loader cannot be
    used for the test set (wrong frame order, asserts one box per frame)."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.anno_files = sorted(glob.glob(os.path.join(root_dir, 'TEST', 'anno', '*.txt')))
        if not self.anno_files:
            sys.exit('TrackingNet test not found (no %s/TEST/anno/*.txt) -> set DATA_ROOTS["TrackingNet"]' % root_dir)
        self.seq_names = [os.path.basename(f)[:-4] for f in self.anno_files]

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.seq_names.index(index)
        d = os.path.join(self.root_dir, 'TEST', 'frames', self.seq_names[index])
        img_files = sorted(glob.glob(os.path.join(d, '*.jpg')), key=lambda f: int(os.path.basename(f)[:-4]))
        anno = np.loadtxt(self.anno_files[index], delimiter=',').reshape(-1, 4)    # first-frame box only
        return img_files, anno

    def __len__(self):
        return len(self.seq_names)


class ExperimentTrackingNet(ExperimentOTB):
    """OPE on the 511 TrackingNet test sequences. No local ground truth: run() records the boxes and
    make_submission() zips them for the official server."""

    def __init__(self, root_dir, result_dir, report_dir):
        self.dataset = TrackingNetTest(root_dir)
        self.result_dir = os.path.join(result_dir, 'TrackingNet')
        self.report_dir = os.path.join(report_dir, 'TrackingNet')
        self.nbins_iou = 21
        self.nbins_ce = 51

    def run(self, tracker, visualize=False):
        print('Running tracker %s on TrackingNet test (%d sequences)...' % (tracker.name, len(self.dataset)))
        for s in range(len(self.dataset)):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))
            record_file = os.path.join(self.result_dir, tracker.name, '%s.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue
            img_files, anno = self.dataset[s]
            boxes, times = tracker.track(img_files, anno[0, :], visualize=visualize)
            assert len(boxes) == len(img_files)
            self._record(record_file, boxes, times)

    def make_submission(self, name):
        import zipfile
        d = os.path.join(self.result_dir, name)
        missing = [s for s in self.dataset.seq_names if not os.path.isfile(os.path.join(d, s + '.txt'))]
        if missing:
            raise RuntimeError('%d sequences without results, e.g. %s' % (len(missing), missing[:3]))
        times = [np.loadtxt(os.path.join(d, 'times', s + '_time.txt')) for s in self.dataset.seq_names]
        fps = float(np.mean([1.0 / np.mean(t[1:]) for t in times if np.size(t) > 1]))   # got10k: mean of seq fps
        zf = os.path.join(self.result_dir, name + '_TrackingNet_submission.zip')
        with zipfile.ZipFile(zf, 'w', zipfile.ZIP_DEFLATED) as z:
            for s in self.dataset.seq_names:
                z.write(os.path.join(d, s + '.txt'), arcname=s + '.txt')
        return zf, fps


def find_otb(hint):
    """OTB100 root = folder that contains Basketball/groundtruth_rect.txt (DTB70 also has a 'Basketball' sequence,
    so only folders whose name contains 'otb' are accepted during the automatic search)."""
    if hint and os.path.isfile(os.path.join(hint, 'Basketball', 'groundtruth_rect.txt')):
        return hint
    for base in ['~/Desktop/testdata', '~/Desktop', '~/data', '~']:
        base = os.path.expanduser(base)
        for p in sorted(glob.glob(os.path.join(base, '*', 'Basketball', 'groundtruth_rect.txt')) +
                        glob.glob(os.path.join(base, '*', '*', 'Basketball', 'groundtruth_rect.txt'))):
            root = os.path.dirname(os.path.dirname(p))
            if 'otb' in os.path.basename(root).lower():
                return root
    sys.exit('OTB100 not found (no */Basketball/groundtruth_rect.txt) -> set DATA_ROOTS["OTB100"] in test.py')


def summarize(e, name):
    perf = e.report([name])[name]['overall']
    return {'AUC': round(100 * perf['success_score'], 1),
            'Prec': round(100 * perf['precision_score'], 1),
            'FPS': round(perf['speed_fps'], 1)}


def make_experiment(bench, root, res_dir, rep_dir):
    if bench == 'OTB100':
        return ExperimentOTB(find_otb(root), version=2015, result_dir=res_dir, report_dir=rep_dir)
    if bench == 'LaSOT':
        return ExperimentLaSOT(root, res_dir, rep_dir)
    if bench == 'UAV123':
        return ExperimentUAV123(root, version='UAV123', result_dir=res_dir, report_dir=rep_dir)
    if bench == 'DTB70':
        return ExperimentDTB70(root, result_dir=res_dir, report_dir=rep_dir)
    if bench == 'TC128':
        return ExperimentTC128(root, res_dir, rep_dir)
    if bench == 'NFS30':
        return ExperimentNfS30(root, res_dir, rep_dir)
    if bench == 'TrackingNet':
        return ExperimentTrackingNet(root, res_dir, rep_dir)
    if bench == 'GOT10k-test':
        return ExperimentGOT10k(root, subset='test', result_dir=res_dir, report_dir=rep_dir)
    raise ValueError('unknown benchmark ' + bench)


def run_benchmark(bench, root, tracker, res_dir, rep_dir, visualize=False):
    e = make_experiment(bench, root, res_dir, rep_dir)
    e.run(tracker, visualize=visualize)
    if bench == 'GOT10k-test':
        e.report([tracker.name])           # zips the results: no ground truth locally, submit the zip online
        return {'submission': os.path.join(res_dir, 'GOT-10k', tracker.name + '.zip')}
    if bench == 'TrackingNet':
        zf, fps = e.make_submission(tracker.name)
        print('TrackingNet submission file (upload it to the official server):', zf)
        return {'submission': zf, 'FPS': round(fps, 1)}
    s = {k: float(v) for k, v in summarize(e, tracker.name).items()}
    if bench == 'LaSOT':
        s['NormPrec'] = round(100 * e.norm_precision(tracker.name), 1)
    return s
