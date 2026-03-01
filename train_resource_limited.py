# -*- coding: utf-8 -*-
"""
train_resource_limited.py
=========================
Training pipeline for the resource-limited ViT (visual2action_3) that predicts
primate directional choices in a Pac-Man junction task.

Three regularization modes are supported:
  none   -- standard training (reproduces baseline)
  l1     -- L1 penalty on MLP weights + proximal gradient (soft-thresholding)
  topk   -- Top-K activation sparsity inside MLP hidden layer

Usage examples:
  # Baseline reproduction
  python train_resource_limited.py --reg none --layers 2 --dim 48 --heads 2

  # L1 regularization sweep
  python train_resource_limited.py --reg l1 --lambda_l1 1e-4 --layers 2 --dim 48 --heads 2

  # Top-K activation sparsity
  python train_resource_limited.py --reg topk --topk_k 8 --layers 2 --dim 48 --heads 2

  # With epoch snapshots every 50 epochs
  python train_resource_limited.py --reg l1 --lambda_l1 1e-4 --snapshot_every 50

Author: Puzhi YU
Date:   January 2026
"""

import argparse
import copy
import importlib.util
import os
import pickle
import sys

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# 0. Import project modules (both have spaces in filename)
# ══════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name, filepath):
    """Load a Python file as a module regardless of spaces in filename."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so that cross-imports (e.g. `from utils import *`) resolve correctly
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


utils = _load_module('utils', os.path.join(_HERE, 'utils.py'))
bb    = _load_module('building_blocks', os.path.join(_HERE, 'building_blocks.py'))

visual2action_3 = bb.visual2action_3


# ══════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════

# Pre-fit label encoder with the known 4 classes (alphabetical → 0-3)
# down=0, left=1, right=2, up=3  (matches model output convention)
_label_enc = LabelEncoder()
_label_enc.fit(['down', 'left', 'right', 'up'])


def load_session(file_path: str):
    """
    Load one session's junction data file (pickle, regardless of .csv/.pkl extension).

    Returns
    -------
    s : np.ndarray  shape (N, 32, 28, 34)  float32  — input frames
    a : np.ndarray  shape (N,)              int64    — action labels (0-3)
    """
    with open(file_path, 'rb') as f:
        tiles, actions, _ = pickle.load(f)

    tiles_cat = np.array([x[0] for x in tiles])   # (N, T=2, H=36, W=28, C=17)
    N, T, H, W, C = tiles_cat.shape

    # Map each cell's 17-channel feature vector to vocabulary indices
    x_flat = torch.tensor(tiles_cat.reshape(-1, C))   # (N*T*H*W, 17)
    props_idx = torch.where(torch.all(utils.v1 == x_flat[:, None, :8],  dim=2))[1]
    chara_idx = torch.where(torch.all(utils.v2 == x_flat[:, None, 8:], dim=2))[1]
    del x_flat

    game_info = np.stack(
        (props_idx.reshape(N, T, H, W).numpy(),
         chara_idx.reshape(N, T, H, W).numpy()),
        axis=-1
    )  # (N, T, H, W, 2)

    # Convert vocab indices → one-hot embedding, crop H from 36→32, flatten time+channel
    obs = utils.label2voc(game_info)                              # (N, T, 36, 28, C_voc)
    s   = einops.rearrange(obs[:, :, 2:34, :, :],
                           'n t h w c -> n h w (t c)')           # (N, 32, 28, 34)
    a   = _label_enc.transform(actions).astype(np.int64)          # (N,)

    return s.astype(np.float32), a


def load_dataset(boarddata_path: str, file_indices: list, desc: str = 'Loading'):
    """
    Load and concatenate multiple sessions.

    Returns
    -------
    S : torch.Tensor  (N_total, 32, 28, 34)
    A : torch.Tensor  (N_total,)  long
    session_boundaries : list of (start, end) tuples  (for per-session eval)
    """
    files = sorted(os.listdir(boarddata_path))
    all_s, all_a, boundaries = [], [], []
    cursor = 0

    for i in tqdm(file_indices, desc=desc, ncols=80):
        path = os.path.join(boarddata_path, files[i])
        s, a = load_session(path)
        all_s.append(s)
        all_a.append(a)
        boundaries.append((cursor, cursor + len(s)))
        cursor += len(s)

    S = torch.tensor(np.concatenate(all_s, axis=0))   # (N, 32, 28, 34)
    A = torch.tensor(np.concatenate(all_a, axis=0))   # (N,)
    return S, A, boundaries


# ══════════════════════════════════════════════════════════════════════
# 2. Resource limitation components
# ══════════════════════════════════════════════════════════════════════

def soft_threshold(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Proximal operator for L1: element-wise soft-thresholding."""
    return torch.sign(tensor) * torch.clamp(tensor.abs() - threshold, min=0.0)


def apply_proximal_l1(model: nn.Module, lambda_l1: float, lr: float):
    """
    After optimizer.step(), apply proximal gradient update to MLP weights.
    This is what makes L1 produce *exact* zeros (unlike plain gradient descent).
    Only targets the MLP (feed-forward) sub-layers inside each encoder Block.
    """
    threshold = lambda_l1 * lr
    for name, param in model.named_parameters():
        if 'encode' in name and 'mlp' in name and 'weight' in name:
            with torch.no_grad():
                param.data = soft_threshold(param.data, threshold)


class TopKMlp(nn.Module):
    """
    Drop-in replacement for timm's Mlp that zeros out all but the top-k
    activations (by absolute value) after the nonlinearity.

    Parameters
    ----------
    k : int   number of hidden neurons to keep active per token per forward pass
    """
    def __init__(self, in_features: int, hidden_features: int,
                 act_layer=nn.GELU, drop: float = 0., k: int = None):
        super().__init__()
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.act   = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2   = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)
        self.k     = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        if self.k is not None and self.k < x.shape[-1]:
            # Build a binary mask that keeps only the top-k activations
            topk_idx = torch.topk(x.abs(), k=self.k, dim=-1).indices
            mask = torch.zeros_like(x).scatter_(-1, topk_idx, 1.0)
            x = x * mask
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def apply_topk_to_model(model: nn.Module, k: int, dim: int) -> nn.Module:
    """Replace every encoder Block's Mlp with TopKMlp."""
    hidden_dim = int(dim * 4)   # matches mlp_ratio=4 in Block
    for block in model.encode:
        block.mlp = TopKMlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.,
            k=k,
        )
    return model


# ══════════════════════════════════════════════════════════════════════
# 3. Logger (matches original utils.logger interface exactly)
# ══════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, items):
        for item in items:
            setattr(self, item, [])

    def update(self, keys, values):
        for k, v in zip(keys, values):
            getattr(self, k).append(v)


# ══════════════════════════════════════════════════════════════════════
# 4. Training / evaluation helpers
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, lambda_l1=0.0):
    """One training epoch. Returns (mean_loss, overall_accuracy)."""
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for s_b, a_b in loader:
        s_b, a_b = s_b.to(device), a_b.to(device)
        optimizer.zero_grad()

        y, loss = model((s_b, a_b))

        # Add L1 penalty on MLP weights to the task loss
        if lambda_l1 > 0:
            l1_term = sum(
                p.abs().sum()
                for name, p in model.named_parameters()
                if 'encode' in name and 'mlp' in name and 'weight' in name
            )
            loss = loss + lambda_l1 * l1_term

        loss.backward()
        optimizer.step()

        # Proximal step AFTER optimizer update (enables exact zeros for L1)
        if lambda_l1 > 0:
            current_lr = optimizer.param_groups[0]['lr']
            apply_proximal_l1(model, lambda_l1, current_lr)

        pred = y.argmax(dim=1)
        total_correct += (pred == a_b).sum().item()
        total_n       += a_b.size(0)
        total_loss    += loss.item() * a_b.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_dataset(model, s: torch.Tensor, a: torch.Tensor, device, batch_size=4096):
    """Evaluate on a full tensor dataset. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    loader = DataLoader(TensorDataset(s, a), batch_size=batch_size, shuffle=False)

    for s_b, a_b in loader:
        s_b, a_b = s_b.to(device), a_b.to(device)
        y, loss  = model((s_b, a_b))
        pred     = y.argmax(dim=1)
        total_correct += (pred == a_b).sum().item()
        total_n       += a_b.size(0)
        total_loss    += loss.item() * a_b.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_per_session(model, s: torch.Tensor, a: torch.Tensor,
                     device, boundaries: list, batch_size=4096) -> list:
    """Return per-session accuracy list (matches original 'train-acc' format)."""
    model.eval()
    accs = []
    for start, end in boundaries:
        s_sess, a_sess = s[start:end], a[start:end]
        loader = DataLoader(TensorDataset(s_sess, a_sess),
                            batch_size=batch_size, shuffle=False)
        correct, n = 0, 0
        for s_b, a_b in loader:
            s_b, a_b = s_b.to(device), a_b.to(device)
            y, _ = model((s_b, a_b))
            correct += (y.argmax(1) == a_b).sum().item()
            n       += a_b.size(0)
        accs.append(correct / n)
    return accs


def mlp_sparsity(model) -> float:
    """Fraction of MLP weights with |w| < 1e-6 (only meaningful for L1 reg)."""
    zeros, total = 0, 0
    for name, p in model.named_parameters():
        if 'encode' in name and 'mlp' in name and 'weight' in name:
            zeros += (p.abs() < 1e-6).sum().item()
            total += p.numel()
    return zeros / total if total > 0 else 0.0


def save_model(path, log, state_dict, configs):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump((log, state_dict, configs), f)


# ══════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train resource-limited transformer for Pac-Man junction prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model architecture
    parser.add_argument('--layers',     type=int,   default=2,
                        help='Number of transformer encoder layers')
    parser.add_argument('--dim',        type=int,   default=12,
                        help='Patch embedding dimension (patch_emb_dim)')
    parser.add_argument('--heads',      type=int,   default=1,
                        help='Number of attention heads')

    # ── Training
    parser.add_argument('--n_epoch',    type=int,   default=800,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int,   default=768)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--earlystop',  type=int,   default=30,
                        help='Stop if test accuracy does not improve for this many epochs')
    parser.add_argument('--seed',       type=int,   default=42)

    # ── Resource limitation
    parser.add_argument('--reg', type=str, default='none',
                        choices=['none', 'l1', 'topk'],
                        help='Regularization / resource-limitation type')
    parser.add_argument('--lambda_l1', type=float, default=1e-4,
                        help='L1 regularization strength λ (used when --reg l1)')
    parser.add_argument('--topk_k',    type=int,   default=8,
                        help='Number of top-k active neurons in MLP (used when --reg topk)')

    # ── Paths
    parser.add_argument('--data_dir', type=str,
                        default='./junction_predict_resortind',
                        help='Directory containing junction pickle files')
    parser.add_argument('--out_dir',  type=str,
                        default='./results',
                        help='Directory to save trained models')

    # ── Misc
    parser.add_argument('--snapshot_every', type=int, default=0,
                        help='Save snapshot every N epochs (0 = disabled)')

    args = parser.parse_args()

    # ── Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Data splits (match original exactly)
    train_indices = list(range(140))
    test_indices  = [140, 141, 142, 143, 144]

    # ── Load data
    print('\n── Loading data ──')
    s_train, a_train, train_boundaries = load_dataset(
        args.data_dir, train_indices, desc='Train')
    s_test,  a_test,  _               = load_dataset(
        args.data_dir, test_indices,  desc='Test')

    print(f'Train: {s_train.shape[0]:,} samples | Test: {s_test.shape[0]:,} samples')

    # ── Build configs (preserve original structure for compatibility)
    configs = {
        # Data / architecture
        'H': 32, 'W': 28,
        't': 2,
        'input_dim': 34,
        'n_layers': args.layers,
        'patch_emb_dim': args.dim,
        'num_heads': args.heads,
        'ker_size': (2, 2),
        'step_size': (2, 2),
        'n_patch': 224,
        'scale_ratio': 4,
        'EmbLN': False,
        'input_scaling': False,
        'vae_attn_drop': 0.0,
        'vae_mlp_drop': 0.0,
        # Training
        'batch_size': args.batch_size,
        'n_epoch': args.n_epoch,
        'earlystop_threshold': args.earlystop,
        'train_files': train_indices,
        'test_files': test_indices,
        'device': device,
        # Paths
        'DataPath': '',
        'boarddata_path': args.data_dir,
        'model_name': 'act_predict_model1',
        'cache_path': '',
        # Resource limitation (new fields)
        'regularization': args.reg,
        'lambda_l1': args.lambda_l1 if args.reg == 'l1'   else 0.0,
        'topk_k':    args.topk_k    if args.reg == 'topk' else None,
        'protocol': 'predict_action',
    }

    # ── Build model
    model = visual2action_3(configs).to(device)

    if args.reg == 'topk':
        model = apply_topk_to_model(model, k=args.topk_k, dim=args.dim)
        model = model.to(device)
        print(f'Top-K sparsity: keeping {args.topk_k} of {args.dim * 4} hidden neurons per MLP')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # ── File naming convention (matches original pattern)
    reg_suffix = {
        'none': '',
        'l1':   f'_L1{args.lambda_l1:.0e}',
        'topk': f'_topk{args.topk_k}',
    }[args.reg]

    model_name     = (f'140sess_{args.layers}layers_focalloss_'
                      f'{args.dim}d{args.heads}h_noEmbLN_'
                      f'{args.batch_size:04d}b{reg_suffix}')
    save_path      = os.path.join(args.out_dir, model_name + '.pkl')
    earlystop_path = os.path.join(args.out_dir, model_name + '_earlystop.pkl')

    # ── Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Logger (same field names as original utils.logger)
    log = Logger(['train_loss', 'test_loss', 'accuracy', 'train-acc'])

    # ── Training loop
    best_test_acc    = -1.0
    best_state_dict  = None
    no_improve_count = 0

    train_loader = DataLoader(
        TensorDataset(s_train, a_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    print(f'\n── Training: {model_name} ──')
    print(f'Max epochs: {args.n_epoch} | Early stop patience: {args.earlystop}')

    for epoch in range(1, args.n_epoch + 1):

        # ── Train
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, device,
            lambda_l1=(args.lambda_l1 if args.reg == 'l1' else 0.0),
        )

        # ── Evaluate on test set (used for early stopping, same as original)
        test_loss, test_acc = eval_dataset(model, s_test, a_test, device)

        # ── Per-session train accuracy (matches original 'train-acc' format)
        per_sess_acc = eval_per_session(
            model, s_train, a_train, device, train_boundaries)

        # ── Log
        log.update(
            ['train_loss', 'test_loss', 'accuracy', 'train-acc'],
            [train_loss,   test_loss,   test_acc,   per_sess_acc],
        )

        # ── Print
        if epoch % 10 == 0 or epoch == 1:
            extra = ''
            if args.reg == 'l1':
                extra = f' | MLP sparsity: {mlp_sparsity(model):.1%}'
            print(f'Epoch {epoch:4d} | '
                  f'train_loss={train_loss:.4f} | '
                  f'test_loss={test_loss:.4f} | '
                  f'test_acc={test_acc:.4f}{extra}')

        # ── Optional epoch snapshots
        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            snap = os.path.join(args.out_dir, f'{model_name}_atEP{epoch:03d}.pkl')
            save_model(snap, log, model.state_dict(), configs)

        # ── Early stopping (based on test accuracy, same as original)
        if test_acc > best_test_acc:
            best_test_acc   = test_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            save_model(earlystop_path, log, best_state_dict, configs)
        else:
            no_improve_count += 1
            if no_improve_count >= args.earlystop:
                print(f'\nEarly stop at epoch {epoch} '
                      f'(best test_acc = {best_test_acc:.4f})')
                break

    # ── Save final model
    save_model(save_path, log, model.state_dict(), configs)

    print(f'\n── Done ──')
    print(f'Best test accuracy : {best_test_acc:.4f}')
    print(f'Final model        : {save_path}')
    print(f'Best checkpoint    : {earlystop_path}')


if __name__ == '__main__':
    main()
