# -*- coding: utf-8 -*-
"""
monitor.py
==========
Quick monitoring script to check experiment progress by reading checkpoint
files. Prints the number of training epochs completed, best accuracy, and
current accuracy for each model in the results directory.

Author: Puzhi YU
Date:   January 2026
"""

import pickle
import glob
import os
import sys
import time

# Allow unpickling of Logger objects from training checkpoints
class Logger:
    pass

import __main__
__main__.Logger = Logger

RESULTS = './results_48d2h'


def check_model(path):
    """Read a checkpoint and return (n_epochs, best_acc, last_acc, tag)."""
    try:
        with open(path, 'rb') as f:
            logger, _, configs = pickle.load(f)
        n = len(logger.accuracy)
        best = max(logger.accuracy)
        last = logger.accuracy[-1]
        reg = configs.get('regularization', configs.get('reg', 'none'))
        lam = configs.get('lambda_l1', '')
        topk = configs.get('topk_k', '')
        tag = f'{reg}'
        if lam:
            tag += f' lam={lam}'
        if topk:
            tag += f' k={topk}'
        return n, best, last, tag
    except Exception:
        return None


print(f'Monitor at {time.strftime("%H:%M:%S")}')
print(f'{"=" * 60}')

files = sorted(glob.glob(os.path.join(RESULTS, '*.pkl')))
if not files:
    print('No checkpoint files found.')
    sys.exit(0)

for f in files:
    name = os.path.basename(f)
    result = check_model(f)
    if result:
        n, best, last, tag = result
        print(f'  {tag:25s} | epochs={n:4d} | best={best:.4f} '
              f'| last={last:.4f} | {name}')
    else:
        print(f'  [error reading] {name}')

print(f'{"=" * 60}')
