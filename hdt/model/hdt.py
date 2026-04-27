"""Hierarchical agent: Transformer planner + MLP executor + Gumbel-Softmax.

Key design: action masking ensures agent never tries to walk into walls.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from config import FEATURE_DIM, NUM_ACTIONS, TrainConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HighLevelNet(nn.Module):
    """Macro planner: Transformer on history → discrete strategy code."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.input_proj = nn.Linear(FEATURE_DIM, config.high_hidden_dim)
        self.pos_enc = PositionalEncoding(config.high_hidden_dim, config.high_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.high_hidden_dim,
            nhead=config.high_num_heads,
            dim_feedforward=config.high_hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.high_num_layers
        )
        self.ln = nn.LayerNorm(config.high_hidden_dim)
        self.code_head = nn.Linear(config.high_hidden_dim, config.num_codes)
        self.value_head = nn.Linear(config.high_hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.ln(x[:, -1])
        return self.code_head(x), self.value_head(x).squeeze(-1)


class LowLevelNet(nn.Module):
    """Micro executor: MLP on current observation + code → action."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        input_dim = FEATURE_DIM + config.num_codes
        hidden = config.low_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x, code_onehot):
        """x: (batch, seq_len, feature_dim) — only last frame used."""
        curr = x[:, -1]  # Current observation
        combined = torch.cat([curr, code_onehot], dim=-1)
        h = self.net(combined)
        return self.action_head(h), self.value_head(h).squeeze(-1)


def _get_action_mask(obs):
    """Extract valid-move mask from observation features.

    Features [30:34] are 1.0 if direction is walkable, 0.0 if wall.
    Returns mask where True = valid action.
    """
    if obs.dim() == 3:
        # (batch, seq_len, feat) -> use last frame
        valid = obs[:, -1, 30:34]  # (batch, 4)
    elif obs.dim() == 2:
        valid = obs[:, 30:34]  # (batch, 4)
    else:
        valid = obs[30:34].unsqueeze(0)
    return valid > 0.5


def _mask_logits(logits, mask):
    """Set logits for invalid actions to -inf."""
    masked = logits.clone()
    masked[~mask] = -1e8
    return masked


class HierarchicalAgent(nn.Module):
    """Combined hierarchical agent with action masking + temporal bottleneck."""

    def __init__(self, config: TrainConfig = None):
        super().__init__()
        self.config = config or TrainConfig()
        self.high_net = HighLevelNet(self.config)
        self.low_net = LowLevelNet(self.config)
        self.num_codes = self.config.num_codes
        self.temperature = self.config.gumbel_temp_start

        self._current_code = None
        self._current_code_onehot = None
        self._steps_since_update = 0

    def update_temperature(self, step: int):
        frac = min(step / max(self.config.gumbel_anneal_steps, 1), 1.0)
        self.temperature = (
            self.config.gumbel_temp_start
            + (self.config.gumbel_temp_end - self.config.gumbel_temp_start) * frac
        )

    def select_code(self, high_seq):
        code_logits, value = self.high_net(high_seq)
        if self.training:
            code_onehot = F.gumbel_softmax(
                code_logits, tau=self.temperature, hard=True
            )
        else:
            code_onehot = F.one_hot(
                code_logits.argmax(-1), self.num_codes
            ).float()
        code_idx = code_onehot.argmax(-1)
        log_prob = F.log_softmax(code_logits, dim=-1)
        log_prob = log_prob.gather(1, code_idx.unsqueeze(1)).squeeze(1)
        return code_idx, code_onehot, log_prob, value

    def select_action(self, low_seq, code_onehot):
        """Sample action with invalid-move masking."""
        action_logits, value = self.low_net(low_seq, code_onehot)
        mask = _get_action_mask(low_seq)
        masked_logits = _mask_logits(action_logits, mask)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_code(self, high_seq, code_idx):
        code_logits, value = self.high_net(high_seq)
        dist = Categorical(logits=code_logits)
        return dist.log_prob(code_idx), value, dist.entropy()

    def evaluate_action(self, low_seq, code_onehot, action):
        """Re-evaluate with masking (must match select_action masking)."""
        action_logits, value = self.low_net(low_seq, code_onehot)
        mask = _get_action_mask(low_seq)
        masked_logits = _mask_logits(action_logits, mask)
        dist = Categorical(logits=masked_logits)
        return dist.log_prob(action), value, dist.entropy()

    @torch.no_grad()
    def act(self, high_seq, low_seq, force_code_update=False):
        """Inference with action masking + temporal bottleneck."""
        self.eval()
        K = self.config.temporal_bottleneck_k

        if (self._current_code_onehot is None or
                self._steps_since_update >= K or
                force_code_update):
            code_logits, _ = self.high_net(high_seq)
            self._current_code = code_logits.argmax(-1).item()
            self._current_code_onehot = F.one_hot(
                torch.tensor([self._current_code], device=high_seq.device),
                self.num_codes
            ).float()
            self._steps_since_update = 0

        action_logits, _ = self.low_net(low_seq, self._current_code_onehot)
        mask = _get_action_mask(low_seq)
        masked_logits = _mask_logits(action_logits, mask)
        # Low-temperature sampling to avoid deterministic loops
        probs = F.softmax(masked_logits / 0.3, dim=-1)
        action = torch.multinomial(probs, 1).item()

        self._steps_since_update += 1
        return self._current_code, action

    def reset_inference(self):
        self._current_code = None
        self._current_code_onehot = None
        self._steps_since_update = 0
