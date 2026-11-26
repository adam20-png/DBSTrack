"""
Train DBSTrack (2.03 M parameters).

    python train.py                  (edit the CONFIGURATION block first)

The published model is trained in two runs:
  1) VARIANT = 'scratch'  : 60 epochs from ImageNet weights
                           phase 1: 10 epochs, MobileNetV3 body FROZEN, new layers lr HEAD_LR[0] -> HEAD_LR[1]
                           phase 2: 50 epochs, everything trained, body lr BACKBONE_LR, rest REST_LR_P2
  2) VARIANT = 'finetune' : 20 more epochs started from the last checkpoint of run 1 (warm restart,
                           lower learning rates)                                  -> 80 epochs in total
  data : GOT-10k train + LaSOT Training + COCO 2017 train (25 % of the pairs of an epoch)
  SGD momentum 0.9, weight decay 1e-4, gradient clipping 10, batch BATCH_SIZE (exponential lr decay)
  loss : focal (cls) + BCE (centerness) + 3 x IoU (boxes)
Validation split (written to models/<RUN_NAME>/val_split.json, used by tune.py only):
  official GOT-10k val (180 sequences) + 20 LaSOT Training sequences that are removed from the training set.

Outputs in models/<RUN_NAME>/ :
  model_e<N>.pth   network weights after epoch N (BN statistics recalibrated), loadable by tune.py / test.py
  last.pth         weights + optimizer + epoch, used to RESUME an interrupted run (just run train.py again)
  val_split.json   the held-out validation sequences
Console log: logs/train_<RUN_NAME>.log

Use a NEW RUN_NAME for every new configuration, otherwise the run resumes from the old last.pth.
"""
from __future__ import absolute_import, print_function

import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from got10k.datasets import GOT10k                                         # noqa: E402
from tracker.backbone import load_imagenet_to_light_backbone                  # noqa: E402
from tracker.coco import CocoPairs                                           # noqa: E402
from tracker.common import init_weights                                      # noqa: E402
from tracker.data import RegPairs                                            # noqa: E402
from tracker.lasot import LaSOTTrain, MixedSeqs                              # noqa: E402
from tracker.model import build_net, reg_loss                               # noqa: E402
from tracker.split import filter_got, filter_lasot, make_split, save_split   # noqa: E402

# ==================================================================== CONFIGURATION — edit, then run
VARIANT = 'scratch'             # 'scratch' (60 epochs from ImageNet) or 'finetune' (20 more epochs)
RUN_NAME = 'dbstrack_e60'       # -> models/<RUN_NAME>/
INIT_FROM = ''                  # checkpoint to start from ('' = ImageNet weights; set below for 'finetune')

# datasets
DATA_GOT = os.path.expanduser('~/data/GOT-10k')             # contains train/ (9335 sequences)
DATA_GOT_VAL = os.path.expanduser('~/data/GOT-10k')         # contains val/ (180 sequences) -> validation only
DATA_LASOT = os.path.expanduser('~/data/LaSOT')             # contains Training/ (and Testing/)
DATA_COCO = os.path.expanduser('~/data/COCO2017')           # contains train2017/ + annotations/
COCO_RATIO = 0.25               # fraction of all pairs of an epoch taken from COCO (0 = no COCO)
GOT_REPEAT = 4                  # pairs per GOT-10k sequence per epoch
LASOT_REPEAT = 16               # pairs per LaSOT sequence per epoch
N_LASOT_VAL = 20                # LaSOT Training sequences held out for validation (never trained on)

# schedule
PHASE1_EPOCHS = 10              # body frozen
PHASE2_EPOCHS = 50              # full network (60 epochs in total for the reference run)
BATCH_SIZE = 8
HEAD_LR = (1e-2, 1e-3)          # phase 1: new layers, start -> end
BACKBONE_LR = (1e-4, 2e-6)      # phase 2: MobileNetV3 body, start -> end
REST_LR_P2 = (5e-4, 1e-5)       # phase 2: all other layers, start -> end
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0

# pair sampling / augmentation
INSTANCE_SZ = 303               # search crop (score map 23 x 23); template 127
MAX_GAPS = (100, 300)           # max frame gap between template and search frame: GOT-10k, LaSOT
NEG_RATIO = 0.2                 # fraction of negative pairs
HARD_NEG = 0.5                  # fraction of negatives taken from the same category
BLUR_PROB = 0.2                 # motion blur on the search crop

# misc
NUM_WORKERS = 10
SEED = 0
SAVE_EVERY = 1                  # save model_e<N>.pth every N epochs (the last epoch is always saved)
SMOKE_TEST = False              # True: 16 GOT + 8 LaSOT sequences + 64 COCO pairs, quick end-to-end check

if VARIANT == 'finetune':       # fine-tuning preset (edit here if needed)
    RUN_NAME = 'dbstrack_ft'
    INIT_FROM = os.path.join(ROOT, 'models', 'dbstrack_e60', 'model_e60.pth')
    PHASE1_EPOCHS = 0           # whole network trained from the first epoch
    PHASE2_EPOCHS = 20
    BACKBONE_LR = (2e-5, 2e-6)
    REST_LR_P2 = (1e-4, 1e-5)
# ====================================================================


class _Tee(object):
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'a', buffering=1)

    def write(self, d):
        self.stream.write(d)
        self.file.write(d)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, n):
        return getattr(self.stream, n)


@torch.no_grad()
def recalibrate_bn(net, loader, dev, n_batches):
    """Exact BN statistics of the new layers for the CURRENT weights before saving (the pretrained body BN stays
    frozen). Without it the running statistics lag behind the weights and tracking degrades strongly."""
    body = set(net.backbone.body.modules())
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m not in body]
    old = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                       # cumulative average
    net.train()
    for i, (z, x, _, _) in enumerate(loader):
        net(z.to(dev, non_blocking=True), x.to(dev, non_blocking=True))
        if i + 1 >= n_batches:
            break
    for m, mo in zip(bns, old):
        m.momentum = mo


def exp_lr(start, end, n, e):
    return start * (end / start) ** (e / max(n - 1, 1))


def build_data(save_dir):
    got = GOT10k(DATA_GOT, subset='train', return_meta=True)
    lasot = LaSOTTrain(DATA_LASOT, subset='Training', return_meta=True)
    split = make_split(got, lasot, n_got=0, n_lasot=N_LASOT_VAL)
    split['got10k_val'] = DATA_GOT_VAL
    save_split(split, os.path.join(save_dir, 'val_split.json'))
    filter_lasot(lasot, split['lasot'], keep=False)
    print('held out for validation: GOT-10k val (%s) + %d LaSOT sequences' % (DATA_GOT_VAL, len(split['lasot'])))
    if SMOKE_TEST:
        for a in ('seq_names', 'seq_dirs', 'anno_files', 'cover_files'):
            if hasattr(got, a):
                setattr(got, a, getattr(got, a)[:16])
        lasot.seq_dirs, lasot.seq_names = lasot.seq_dirs[:8], lasot.seq_names[:8]
    seqs = MixedSeqs([got, lasot], repeats=[GOT_REPEAT, LASOT_REPEAT], names=['GOT-10k', 'LaSOT'])
    print('Training data:')
    print(seqs.summary())
    video = RegPairs(seqs, neg_ratio=NEG_RATIO, instance_sz=INSTANCE_SZ, max_gaps=list(MAX_GAPS),
                     hard_neg=HARD_NEG, blur_prob=BLUR_PROB)
    if COCO_RATIO <= 0:
        return video
    n_coco = 64 if SMOKE_TEST else int(round(len(video) * COCO_RATIO / (1.0 - COCO_RATIO)))
    coco = CocoPairs(DATA_COCO, n_coco, neg_ratio=NEG_RATIO, hard_neg=HARD_NEG, instance_sz=INSTANCE_SZ,
                     blur_prob=BLUR_PROB)
    print(coco.summary())
    print('  total   : %6d pairs/epoch (%.1f %% COCO)' % (len(video) + len(coco),
                                                        100.0 * len(coco) / (len(video) + len(coco))))
    return ConcatDataset([video, coco])


def make_optimizer(phase1, body, rest):
    if phase1:
        return torch.optim.SGD(rest, lr=HEAD_LR[0], momentum=0.9, weight_decay=WEIGHT_DECAY)
    return torch.optim.SGD([{'params': body, 'lr': BACKBONE_LR[0]}, {'params': rest, 'lr': REST_LR_P2[0]}],
                           momentum=0.9, weight_decay=WEIGHT_DECAY)


def learning_rates(epoch):
    if epoch < PHASE1_EPOCHS:
        return [exp_lr(HEAD_LR[0], HEAD_LR[1], PHASE1_EPOCHS, epoch)]
    e = epoch - PHASE1_EPOCHS
    return [exp_lr(BACKBONE_LR[0], BACKBONE_LR[1], PHASE2_EPOCHS, e),
            exp_lr(REST_LR_P2[0], REST_LR_P2[1], PHASE2_EPOCHS, e)]


def main():
    save_dir = os.path.join(ROOT, 'models', RUN_NAME + ('_smoke' if SMOKE_TEST else ''))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    log = os.path.join(ROOT, 'logs', 'train_%s.log' % os.path.basename(save_dir))
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total = PHASE1_EPOCHS + PHASE2_EPOCHS
    print('=' * 78)
    print('DBSTrack (%s) training | run %s | torch %s | device %s%s' % (
        VARIANT, RUN_NAME, torch.__version__, dev, ' (%s)' % torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''))
    print('phase 1 %d ep (body frozen) | phase 2 %d ep | batch %d | search %d'
          % (PHASE1_EPOCHS, PHASE2_EPOCHS, BATCH_SIZE, INSTANCE_SZ))
    print('=' * 78)
    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple, bool))}
    json.dump(config, open(os.path.join(save_dir, 'train_config.json'), 'w'), indent=1)

    net = build_net('dbstrack')
    body = list(net.backbone.body.parameters())
    body_ids = {id(p) for p in body}
    rest = [p for p in net.parameters() if id(p) not in body_ids]

    start_epoch, resume = 0, None
    last = os.path.join(save_dir, 'last.pth')
    if os.path.isfile(last):
        resume = torch.load(last, map_location='cpu')
        net.load_state_dict(resume['net'])
        start_epoch = resume['epoch'] + 1
        print('RESUME from %s (epoch %d done)' % (last, resume['epoch'] + 1))
    elif INIT_FROM:
        sd = torch.load(INIT_FROM, map_location='cpu')
        res = net.load_state_dict(sd.get('net', sd), strict=False)
        assert not res.missing_keys and not res.unexpected_keys, res
        print('initialised from %s' % INIT_FROM)
    else:
        init_weights(net)
        net.head.reset_parameters()
        load_imagenet_to_light_backbone(net, 'IMAGENET1K_V2')
    net.to(dev)
    print('parameters: %.3f M' % (sum(p.numel() for p in net.parameters()) / 1e6))
    if start_epoch >= total:
        print('nothing to do: %d epochs already done (increase PHASE2_EPOCHS or change RUN_NAME)' % start_epoch)
        return

    dataset = build_data(save_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                        pin_memory=dev.type == 'cuda', drop_last=True, persistent_workers=NUM_WORKERS > 0,
                        worker_init_fn=lambda w: np.random.seed((SEED * 1000 + w + int(time.time())) % 2 ** 31))

    opt = make_optimizer(start_epoch < PHASE1_EPOCHS, body, rest)
    if resume is not None and (resume['epoch'] < PHASE1_EPOCHS) == (start_epoch < PHASE1_EPOCHS):
        opt.load_state_dict(resume['opt'])        # same phase -> keep the momentum buffers
    for epoch in range(start_epoch, total):
        phase1 = epoch < PHASE1_EPOCHS
        if epoch == PHASE1_EPOCHS and epoch != start_epoch:
            opt = make_optimizer(False, body, rest)
        for p in body:
            p.requires_grad = not phase1
        lrs = learning_rates(epoch)
        for g, lr in zip(opt.param_groups, lrs):
            g['lr'] = lr

        net.train()
        t0 = time.time()
        meter = {}
        for it, (z, x, box, neg) in enumerate(loader):
            z, x, box, neg = z.to(dev, non_blocking=True), x.to(dev, non_blocking=True), box.to(dev), neg.to(dev)
            loss, parts = reg_loss(net(z, x), box, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            for k, v in parts.items():
                meter[k] = meter[k] * 0.98 + v * 0.02 if k in meter else v
            if it % 20 == 0 or it == len(loader) - 1:
                print('[%s] Epoch: %d/%d [%d/%d] Loss: %.4f | cls %.3f ctr %.3f reg %.3f IoU %.3f pos/img %.1f | lr %.1e'
                      % ('Phase1' if phase1 else 'Phase2', epoch + 1, total, it + 1, len(loader),
                         loss.item(), meter['cls'], meter['ctr'], meter['reg'], meter['iou'],
                         meter['npos'], lrs[-1]))
                sys.stdout.flush()
        recalibrate_bn(net, loader, dev, n_batches=max(60, 960 // BATCH_SIZE))
        if (epoch + 1) % SAVE_EVERY == 0 or epoch + 1 == total:
            path = os.path.join(save_dir, 'model_e%d.pth' % (epoch + 1))
            torch.save(net.state_dict(), path)
            print('saved', path)
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, last + '.tmp')
        os.replace(last + '.tmp', last)
        print('epoch %d done in %.0fs' % (epoch + 1, time.time() - t0))
    print('\nDone! Checkpoints saved to:', save_dir)


if __name__ == '__main__':
    main()
