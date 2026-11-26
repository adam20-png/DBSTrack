"""
Held-out VALIDATION split carved from the training sets (no GOT-10k val available).

Deterministic: every k-th sequence (sorted by name) of GOT-10k train and LaSOT Training is held out.
These sequences are NEVER used for training; they are used ONLY to tune the tracker hyper-parameters
(penalty_k, window_influence, test_lr). OTB100 / LaSOT test are never used for any choice.
"""
from __future__ import absolute_import

import json
import os


def make_split(got, lasot, n_got=120, n_lasot=20):
    g = sorted(got.seq_names)
    l_ = sorted(lasot.seq_names)
    n_got = min(n_got, len(g) // 10)          # never more than 10 % of a set (tiny test sets)
    n_lasot = min(n_lasot, len(l_) // 10)
    val_got = g[::max(1, len(g) // max(n_got, 1))][:n_got] if n_got > 0 else []
    val_lasot = l_[::max(1, len(l_) // max(n_lasot, 1))][:n_lasot]
    return {'got10k': val_got, 'lasot': val_lasot}


def save_split(split, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split, f, indent=1)


def load_split(path):
    with open(path) as f:
        return json.load(f)


def filter_got(got, names, keep):
    """keep=True -> only `names`; keep=False -> everything except `names` (in place)."""
    names = set(names)
    idx = [i for i, n in enumerate(got.seq_names) if (n in names) == keep]
    got.seq_names = [got.seq_names[i] for i in idx]
    got.seq_dirs = [got.seq_dirs[i] for i in idx]
    got.anno_files = [got.anno_files[i] for i in idx]
    return got


def filter_lasot(lasot, names, keep):
    names = set(names)
    idx = [i for i, n in enumerate(lasot.seq_names) if (n in names) == keep]
    lasot.seq_names = [lasot.seq_names[i] for i in idx]
    lasot.seq_dirs = [lasot.seq_dirs[i] for i in idx]
    return lasot
