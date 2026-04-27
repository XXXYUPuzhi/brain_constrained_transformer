"""PPO training with curriculum learning for Hierarchical Decision Transformer."""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from config import TrainConfig, CURRICULUM, FEATURE_DIM, CurriculumStage
from environment.pacman_env import PacmanEnv
from model.hdt import HierarchicalAgent


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Compute GAE advantages and returns."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            nv = next_value
        else:
            nv = values[t + 1]
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nd - values[t]
        gae = delta + gamma * lam * nd * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def evaluate_agent(agent, config, stage, num_episodes=20, device='cpu'):
    """Evaluate agent's win rate."""
    env = PacmanEnv(config, stage)
    wins = 0
    total_score = 0
    total_pellet_ratio = 0.0
    was_training = agent.training
    agent.eval()

    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset_inference()
        high_buf = deque(maxlen=config.high_seq_len)
        low_buf = deque(maxlen=config.low_seq_len)
        zero = np.zeros(FEATURE_DIM, dtype=np.float32)
        for _ in range(config.high_seq_len):
            high_buf.append(zero.copy())
        for _ in range(config.low_seq_len):
            low_buf.append(zero.copy())

        for step in range(stage.max_steps):
            high_buf.append(obs)
            low_buf.append(obs)
            hs = torch.FloatTensor(np.array(list(high_buf))).unsqueeze(0).to(device)
            ls = torch.FloatTensor(np.array(list(low_buf))).unsqueeze(0).to(device)
            _, action = agent.act(hs, ls)
            obs, _, done, info = env.step(action)
            if done:
                break

        if info.get("win", False):
            wins += 1
        total_score += info.get("score", 0)
        # Track pellet ratio
        pe = info.get("pellets_eaten", 0)
        tp = info.get("total_pellets", 200)
        total_pellet_ratio += pe / max(tp, 1)

    if was_training:
        agent.train()
    avg_pellet_ratio = total_pellet_ratio / num_episodes
    return avg_pellet_ratio, total_score / num_episodes


class EnvWorker:
    """Manages one environment with its history buffers."""

    def __init__(self, config, stage, device):
        self.config = config
        self.device = device
        self.env = PacmanEnv(config, stage)
        self.reset()

    def set_stage(self, stage):
        self.env.set_stage(stage)
        self.reset()

    def reset(self):
        self.obs = self.env.reset()
        zero = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.high_buf = deque(maxlen=self.config.high_seq_len)
        self.low_buf = deque(maxlen=self.config.low_seq_len)
        for _ in range(self.config.high_seq_len):
            self.high_buf.append(zero.copy())
        for _ in range(self.config.low_seq_len):
            self.low_buf.append(zero.copy())
        self.current_code_onehot = None
        self.steps_since_code = 0
        self.episode_reward = 0.0

    def get_seqs(self):
        self.high_buf.append(self.obs)
        self.low_buf.append(self.obs)
        hs = torch.FloatTensor(np.array(list(self.high_buf))).unsqueeze(0).to(self.device)
        ls = torch.FloatTensor(np.array(list(self.low_buf))).unsqueeze(0).to(self.device)
        return hs, ls

    def needs_code(self):
        return (self.current_code_onehot is None or
                self.steps_since_code >= self.config.temporal_bottleneck_k)


def collect_rollout_per_env(agent, worker, config, device):
    """Collect a rollout from a single environment. Returns per-env buffers.

    This avoids the interleaving bug by collecting full rollout per env.
    """
    low_seqs = []
    code_onehots = []
    actions = []
    low_log_probs = []
    low_values = []
    rewards = []
    dones = []

    high_seqs = []
    code_indices = []
    high_log_probs = []
    high_values = []
    high_rewards = []
    high_dones = []
    high_reward_accum = 0.0

    episode_scores = []
    episode_wins = []

    for step in range(config.rollout_length):
        hs, ls = worker.get_seqs()

        # High-level decision
        if worker.needs_code():
            # Store previous segment's reward
            if worker.current_code_onehot is not None and len(high_seqs) > 0:
                high_rewards.append(high_reward_accum)
                high_dones.append(False)

            code_idx, code_oh, h_lp, h_val = agent.select_code(hs)
            worker.current_code_onehot = code_oh.detach()
            worker.steps_since_code = 0
            high_reward_accum = 0.0

            high_seqs.append(hs.squeeze(0).detach().cpu())
            code_indices.append(code_idx.item())
            high_log_probs.append(h_lp.item())
            high_values.append(h_val.item())

        # Low-level action
        action, l_lp, l_val, l_ent = agent.select_action(
            ls, worker.current_code_onehot
        )

        low_seqs.append(ls.squeeze(0).detach().cpu())
        code_onehots.append(worker.current_code_onehot.squeeze(0).detach().cpu())
        actions.append(action.item())
        low_log_probs.append(l_lp.item())
        low_values.append(l_val.item())

        # Step env
        next_obs, reward, done, info = worker.env.step(action.item())
        rewards.append(reward)
        dones.append(float(done))
        high_reward_accum += reward
        worker.steps_since_code += 1
        worker.episode_reward += reward

        if done:
            episode_scores.append(info.get("score", 0))
            episode_wins.append(1.0 if info.get("win", False) else 0.0)

            # Close high-level segment
            if len(high_seqs) > len(high_rewards):
                high_rewards.append(high_reward_accum)
                high_dones.append(True)

            worker.obs = worker.env.reset()
            zero = np.zeros(FEATURE_DIM, dtype=np.float32)
            worker.high_buf.clear()
            worker.low_buf.clear()
            for _ in range(config.high_seq_len):
                worker.high_buf.append(zero.copy())
            for _ in range(config.low_seq_len):
                worker.low_buf.append(zero.copy())
            worker.current_code_onehot = None
            worker.steps_since_code = 0
            worker.episode_reward = 0.0
            high_reward_accum = 0.0
        else:
            worker.obs = next_obs

    # Close last high-level segment if needed
    if len(high_seqs) > len(high_rewards):
        high_rewards.append(high_reward_accum)
        high_dones.append(False)

    # Get bootstrap values
    with torch.no_grad():
        hs, ls = worker.get_seqs()
        _, low_next_val = agent.low_net(
            ls, worker.current_code_onehot if worker.current_code_onehot is not None
            else torch.zeros(1, config.num_codes, device=device)
        )
        _, high_next_val = agent.high_net(hs)

    return {
        'low_seqs': low_seqs, 'code_onehots': code_onehots,
        'actions': actions, 'low_log_probs': low_log_probs,
        'low_values': low_values, 'rewards': rewards, 'dones': dones,
        'low_next_value': low_next_val.item(),
        'high_seqs': high_seqs, 'code_indices': code_indices,
        'high_log_probs': high_log_probs, 'high_values': high_values,
        'high_rewards': high_rewards, 'high_dones': high_dones,
        'high_next_value': high_next_val.item(),
        'episode_scores': episode_scores, 'episode_wins': episode_wins,
    }


def ppo_update(agent, optimizer, all_rollouts, config, device):
    """Run PPO update on collected rollouts."""
    # Merge low-level data (now properly separated per-env, GAE computed per-env)
    all_low_seqs = []
    all_code_ohs = []
    all_actions = []
    all_old_lps = []
    all_low_advs = []
    all_low_rets = []

    all_high_seqs = []
    all_codes = []
    all_old_h_lps = []
    all_high_advs = []
    all_high_rets = []

    for rollout in all_rollouts:
        # Low-level GAE (per env, no interleaving!)
        advs, rets = compute_gae(
            rollout['rewards'], rollout['low_values'], rollout['dones'],
            rollout['low_next_value'], config.gamma, config.gae_lambda,
        )
        all_low_seqs.extend(rollout['low_seqs'])
        all_code_ohs.extend(rollout['code_onehots'])
        all_actions.extend(rollout['actions'])
        all_old_lps.extend(rollout['low_log_probs'])
        all_low_advs.append(advs)
        all_low_rets.append(rets)

        # High-level GAE
        n_h = min(len(rollout['high_seqs']), len(rollout['high_rewards']),
                  len(rollout['high_values']))
        if n_h > 1:
            h_advs, h_rets = compute_gae(
                rollout['high_rewards'][:n_h],
                rollout['high_values'][:n_h],
                rollout['high_dones'][:n_h],
                rollout['high_next_value'],
                config.gamma, config.gae_lambda,
            )
            all_high_seqs.extend(rollout['high_seqs'][:n_h])
            all_codes.extend(rollout['code_indices'][:n_h])
            all_old_h_lps.extend(rollout['high_log_probs'][:n_h])
            all_high_advs.append(h_advs)
            all_high_rets.append(h_rets)

    # Convert to tensors
    n_low = len(all_actions)
    low_seqs_t = torch.stack(all_low_seqs)
    code_oh_t = torch.stack(all_code_ohs)
    actions_t = torch.tensor(all_actions, dtype=torch.long)
    old_low_lp = torch.tensor(all_old_lps, dtype=torch.float32)
    low_advs = torch.from_numpy(np.concatenate(all_low_advs))
    low_advs = (low_advs - low_advs.mean()) / (low_advs.std() + 1e-8)
    low_rets = torch.from_numpy(np.concatenate(all_low_rets))

    n_high = len(all_codes)
    has_high = n_high > 1
    if has_high:
        high_seqs_t = torch.stack(all_high_seqs)
        codes_t = torch.tensor(all_codes, dtype=torch.long)
        old_high_lp = torch.tensor(all_old_h_lps, dtype=torch.float32)
        high_advs = torch.from_numpy(np.concatenate(all_high_advs))
        high_advs = (high_advs - high_advs.mean()) / (high_advs.std() + 1e-8)
        high_rets = torch.from_numpy(np.concatenate(all_high_rets))

    # PPO epochs
    total_low_loss = 0
    total_high_loss = 0
    for _ in range(config.ppo_epochs):
        perm = torch.randperm(n_low)
        for start in range(0, n_low, config.mini_batch_size):
            end = min(start + config.mini_batch_size, n_low)
            idx = perm[start:end]

            # Low-level PPO
            mb_ls = low_seqs_t[idx].to(device)
            mb_co = code_oh_t[idx].to(device)
            mb_a = actions_t[idx].to(device)
            mb_olp = old_low_lp[idx].to(device)
            mb_adv = low_advs[idx].to(device)
            mb_ret = low_rets[idx].to(device)

            new_lp, new_val, new_ent = agent.evaluate_action(mb_ls, mb_co, mb_a)
            ratio = torch.exp(new_lp - mb_olp)
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * mb_adv
            p_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(new_val, mb_ret)
            e_loss = -new_ent.mean()
            low_loss = p_loss + config.value_loss_coef * v_loss + config.entropy_coef * e_loss

            # High-level PPO
            high_loss = torch.tensor(0.0, device=device)
            if has_high:
                h_size = min(end - start, n_high)
                h_idx = torch.randint(0, n_high, (h_size,))
                mb_hs = high_seqs_t[h_idx].to(device)
                mb_c = codes_t[h_idx].to(device)
                mb_holp = old_high_lp[h_idx].to(device)
                mb_hadv = high_advs[h_idx].to(device)
                mb_hret = high_rets[h_idx].to(device)

                h_new_lp, h_new_val, h_new_ent = agent.evaluate_code(mb_hs, mb_c)
                h_ratio = torch.exp(h_new_lp - mb_holp)
                hs1 = h_ratio * mb_hadv
                hs2 = torch.clamp(h_ratio, 1 - config.clip_ratio,
                                  1 + config.clip_ratio) * mb_hadv
                h_p_loss = -torch.min(hs1, hs2).mean()
                h_v_loss = F.mse_loss(h_new_val, mb_hret)
                h_e_loss = -h_new_ent.mean()

                # Code diversity loss: penalize if batch uses same codes
                # Ideal: uniform distribution over codes → max entropy
                code_counts = torch.bincount(mb_c, minlength=config.num_codes).float()
                code_dist = code_counts / code_counts.sum()
                uniform = torch.ones_like(code_dist) / config.num_codes
                diversity_loss = F.kl_div(
                    (code_dist + 1e-8).log(), uniform, reduction='sum'
                )

                high_loss = (h_p_loss
                             + config.value_loss_coef * h_v_loss
                             + config.high_entropy_coef * h_e_loss
                             + 0.1 * diversity_loss)

            loss = low_loss + high_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()

            total_low_loss += low_loss.item()
            total_high_loss += high_loss.item()

    n_updates = max(config.ppo_epochs * (n_low // config.mini_batch_size + 1), 1)
    return total_low_loss / n_updates, total_high_loss / n_updates


def train(config: TrainConfig = None):
    config = config or TrainConfig()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    print(f"Seed: {config.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    agent = HierarchicalAgent(config).to(device)
    param_count = sum(p.numel() for p in agent.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Curriculum
    current_stage_idx = 0
    stage = CURRICULUM[current_stage_idx]
    print(f"\n>>> Starting {stage.name} (ghosts={stage.num_ghosts}, "
          f"power={stage.has_power_pellets})")

    workers = [EnvWorker(config, stage, device) for _ in range(config.num_envs)]

    episode_scores = deque(maxlen=100)
    episode_wins = deque(maxlen=100)
    total_steps = 0
    start_time = time.time()

    while total_steps < config.total_timesteps:
        agent.train()

        # Collect rollouts per-env (no interleaving!)
        all_rollouts = []
        for worker in workers:
            rollout = collect_rollout_per_env(agent, worker, config, device)
            all_rollouts.append(rollout)
            for s in rollout['episode_scores']:
                episode_scores.append(s)
            for w in rollout['episode_wins']:
                episode_wins.append(w)

        total_steps += config.rollout_length * config.num_envs
        agent.update_temperature(total_steps)

        # PPO update
        low_loss, high_loss = ppo_update(agent, optimizer, all_rollouts, config, device)

        # Logging
        if total_steps % config.log_interval < config.rollout_length * config.num_envs:
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1)
            avg_score = np.mean(episode_scores) if episode_scores else 0
            win_rate = np.mean(episode_wins) if episode_wins else 0
            print(f"[{total_steps:>8d}] {stage.name} "
                  f"score={avg_score:.0f} win={win_rate:.0%} "
                  f"low_loss={low_loss:.3f} high_loss={high_loss:.3f} "
                  f"temp={agent.temperature:.2f} fps={fps:.0f}")
            sys.stdout.flush()

        # Curriculum eval
        if total_steps % config.eval_interval < config.rollout_length * config.num_envs:
            pellet_ratio, avg_score = evaluate_agent(
                agent, config, stage,
                num_episodes=stage.eval_episodes, device=device,
            )
            print(f"  [EVAL] pellet_ratio={pellet_ratio:.0%} score={avg_score:.0f} "
                  f"(need {stage.win_threshold:.0%})")
            sys.stdout.flush()

            if pellet_ratio >= stage.win_threshold:
                # Save stage checkpoint
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'config': config,
                    'stage_idx': current_stage_idx,
                    'total_steps': total_steps,
                }, os.path.join(save_dir, f"stage{current_stage_idx}_done.pt"))

                current_stage_idx += 1
                if current_stage_idx >= len(CURRICULUM):
                    print(f"\n>>> ALL STAGES COMPLETE at step {total_steps}!")
                    torch.save({
                        'model_state_dict': agent.state_dict(),
                        'config': config,
                        'stage': 'complete',
                        'total_steps': total_steps,
                    }, os.path.join(save_dir, "final_model.pt"))
                    break

                stage = CURRICULUM[current_stage_idx]
                print(f"\n>>> Advancing to {stage.name} "
                      f"(ghosts={stage.num_ghosts}, power={stage.has_power_pellets})")
                for w in workers:
                    w.set_stage(stage)
                episode_scores.clear()
                episode_wins.clear()

        # Periodic save
        if total_steps % config.save_interval < config.rollout_length * config.num_envs:
            torch.save({
                'model_state_dict': agent.state_dict(),
                'config': config,
                'stage_idx': current_stage_idx,
                'total_steps': total_steps,
            }, os.path.join(save_dir, "latest.pt"))

    elapsed = time.time() - start_time
    print(f"\nDone. {total_steps:,} steps in {elapsed/60:.1f} min")


if __name__ == "__main__":
    config = TrainConfig()
    if len(sys.argv) > 1:
        config.total_timesteps = int(sys.argv[1])
    train(config)
