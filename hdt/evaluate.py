"""Evaluate trained HDT model and visualize emergent strategy codes."""
import os
import sys
import time
import torch
import numpy as np
from collections import deque, defaultdict
from config import TrainConfig, CURRICULUM, FEATURE_DIM, ACTION_NAMES
from environment.pacman_env import PacmanEnv
from model.hdt import HierarchicalAgent


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    agent = HierarchicalAgent(config).to(device)
    agent.load_state_dict(ckpt['model_state_dict'])
    agent.eval()
    print(f"Loaded model (steps: {ckpt.get('total_steps', '?')})")
    return agent, config


def analyze_codes(agent, config, stage, num_episodes=50, device='cpu'):
    """Analyze what each latent code corresponds to in game context.

    For each code, collects statistics about:
    - Ghost proximity when code is active
    - Whether ghosts are frightened
    - Pellet density around pacman
    - Outcomes (eating pellets, eating ghosts, dying)
    """
    env = PacmanEnv(config, stage)
    K = config.temporal_bottleneck_k

    # Per-code statistics
    code_stats = defaultdict(lambda: {
        'count': 0,
        'avg_ghost_dist': [],
        'ghost_frightened_pct': [],
        'pellets_eaten': 0,
        'ghosts_eaten': 0,
        'deaths': 0,
        'power_pellet_nearby': [],
        'avg_remaining_pellets': [],
    })

    for ep in range(num_episodes):
        obs = env.reset()
        agent.reset_inference()
        high_history = deque(maxlen=config.high_seq_len)
        low_history = deque(maxlen=config.low_seq_len)
        for _ in range(config.high_seq_len):
            high_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        for _ in range(config.low_seq_len):
            low_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        current_code = None
        prev_pellets = env.pellets_eaten

        for step in range(stage.max_steps):
            high_history.append(obs)
            low_history.append(obs)

            high_seq = torch.FloatTensor(
                np.array(list(high_history))
            ).unsqueeze(0).to(device)
            low_seq = torch.FloatTensor(
                np.array(list(low_history))
            ).unsqueeze(0).to(device)

            code, action = agent.act(high_seq, low_seq)
            current_code = code

            # Record context for this code
            stats = code_stats[code]
            stats['count'] += 1

            # Ghost distance (from feature vector)
            ghost_dists = []
            any_frightened = False
            for i in range(4):
                base = 2 + i * 5
                if obs[base + 4] > 0.5:  # is_active
                    ghost_dists.append(obs[base + 2])  # normalized distance
                    if obs[base + 3] > 0.5:  # is_frightened
                        any_frightened = True
            if ghost_dists:
                stats['avg_ghost_dist'].append(np.mean(ghost_dists))
            stats['ghost_frightened_pct'].append(1.0 if any_frightened else 0.0)
            stats['power_pellet_nearby'].append(1.0 if obs[27] > 0 else 0.0)
            stats['avg_remaining_pellets'].append(obs[28])

            obs, reward, done, info = env.step(action)

            # Track events
            if env.pellets_eaten > prev_pellets:
                stats['pellets_eaten'] += (env.pellets_eaten - prev_pellets)
            prev_pellets = env.pellets_eaten

            if info.get('collision') == 'eat_ghost':
                stats['ghosts_eaten'] += 1
            elif info.get('collision') in ('death', 'game_over'):
                stats['deaths'] += 1

            if done:
                break

    # Print analysis
    print(f"\n{'='*70}")
    print(f"EMERGENT STRATEGY CODE ANALYSIS ({stage.name}, {num_episodes} episodes)")
    print(f"{'='*70}")

    for code_id in sorted(code_stats.keys()):
        stats = code_stats[code_id]
        n = stats['count']
        if n == 0:
            continue

        avg_gdist = np.mean(stats['avg_ghost_dist']) if stats['avg_ghost_dist'] else -1
        fright_pct = np.mean(stats['ghost_frightened_pct']) * 100
        pp_nearby = np.mean(stats['power_pellet_nearby']) * 100
        rem_pellets = np.mean(stats['avg_remaining_pellets']) * 100

        # Infer strategy label
        label = "???"
        if avg_gdist < 0:
            label = "FORAGE (no ghosts)"
        elif fright_pct > 40:
            label = "HUNT (ghosts frightened)"
        elif avg_gdist < 0.15:
            label = "EVADE (ghosts close)"
        elif pp_nearby > 50 and avg_gdist < 0.3:
            label = "SEEK POWER"
        elif avg_gdist >= 0.15:
            label = "FORAGE (safe)"
        else:
            label = "MIXED"

        print(f"\nCode {code_id} [{label}]  (activated {n} times, "
              f"{100*n/sum(s['count'] for s in code_stats.values()):.1f}%)")
        print(f"  Avg ghost distance:    {avg_gdist:.3f} "
              f"{'(CLOSE!)' if 0 < avg_gdist < 0.15 else ''}")
        print(f"  Ghosts frightened:     {fright_pct:.1f}%")
        print(f"  Power pellet nearby:   {pp_nearby:.1f}%")
        print(f"  Remaining pellets:     {rem_pellets:.0f}%")
        print(f"  Pellets eaten:         {stats['pellets_eaten']}")
        print(f"  Ghosts eaten:          {stats['ghosts_eaten']}")
        print(f"  Deaths:                {stats['deaths']}")

    return code_stats


def evaluate(agent, config, stage, num_episodes=100, device='cpu'):
    """Evaluate agent performance."""
    env = PacmanEnv(config, stage)
    wins = 0
    total_score = 0
    ghost_kills = 0

    for ep in range(num_episodes):
        obs = env.reset()
        agent.reset_inference()
        high_history = deque(maxlen=config.high_seq_len)
        low_history = deque(maxlen=config.low_seq_len)
        for _ in range(config.high_seq_len):
            high_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        for _ in range(config.low_seq_len):
            low_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        for step in range(stage.max_steps):
            high_history.append(obs)
            low_history.append(obs)
            high_seq = torch.FloatTensor(
                np.array(list(high_history))
            ).unsqueeze(0).to(device)
            low_seq = torch.FloatTensor(
                np.array(list(low_history))
            ).unsqueeze(0).to(device)

            _, action = agent.act(high_seq, low_seq)
            obs, _, done, info = env.step(action)
            if info.get('collision') == 'eat_ghost':
                ghost_kills += 1
            if done:
                break

        if info.get('win', False):
            wins += 1
        total_score += info.get('score', 0)

    print(f"\n{'='*50}")
    print(f"EVALUATION ({stage.name}, {num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Win rate:     {100*wins/num_episodes:.1f}%")
    print(f"Avg score:    {total_score/num_episodes:.1f}")
    print(f"Ghost kills:  {ghost_kills}")


def demo(agent, config, stage, device='cpu'):
    """Run one game with ASCII rendering and code visualization."""
    env = PacmanEnv(config, stage)
    obs = env.reset()
    agent.reset_inference()
    high_history = deque(maxlen=config.high_seq_len)
    low_history = deque(maxlen=config.low_seq_len)
    for _ in range(config.high_seq_len):
        high_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    for _ in range(config.low_seq_len):
        low_history.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    print("Running demo (Ctrl+C to stop)...\n")
    prev_code = None
    try:
        for step in range(stage.max_steps):
            high_history.append(obs)
            low_history.append(obs)
            high_seq = torch.FloatTensor(
                np.array(list(high_history))
            ).unsqueeze(0).to(device)
            low_seq = torch.FloatTensor(
                np.array(list(low_history))
            ).unsqueeze(0).to(device)

            code, action = agent.act(high_seq, low_seq)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(env.render_ascii())
            print(f"\nStep: {step}  Score: {env.score}  Lives: {env.lives}")
            print(f"Active Code: {code}  Action: {ACTION_NAMES[action]}")

            if prev_code is not None and code != prev_code:
                print(f"  >>> CODE SWITCH: {prev_code} -> {code}")
            prev_code = code

            obs, _, done, info = env.step(action)
            if done:
                result = "WIN!" if info.get('win') else "GAME OVER"
                print(f"\n{result}  Final score: {env.score}")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    # Try loading best available checkpoint
    for name in ["final_model.pt", "latest.pt"]:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            agent, config = load_model(path, device)
            break
    else:
        print("No checkpoint found. Run train.py first.")
        sys.exit(1)

    # Use final curriculum stage
    stage = CURRICULUM[-1]

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "demo":
        demo(agent, config, stage, device)
    elif mode == "analyze":
        analyze_codes(agent, config, stage, num_episodes=50, device=device)
    else:  # "all"
        evaluate(agent, config, stage, num_episodes=100, device=device)
        analyze_codes(agent, config, stage, num_episodes=50, device=device)
