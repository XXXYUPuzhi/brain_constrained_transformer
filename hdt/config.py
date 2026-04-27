"""Configuration for PacmanHDT — Classic Arcade Layout + Juice-Drop Rewards."""
from dataclasses import dataclass

# Cell types
WALL = 0
EMPTY = 1
PELLET = 2
POWER_PELLET = 3
GHOST_DOOR = 4
GHOST_HOUSE = 5

# Actions (no STAY)
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_ACTIONS = 4

# Ghost modes
CHASE = 0
SCATTER = 1
FRIGHTENED = 2

# Feature dimensions (30 base + 4 valid_moves + 4 pellet_adjacent + 4 corridor = 42)
FEATURE_DIM = 42

# Classic Pac-Man arcade maze (31 rows x 28 cols)
# '#' = wall, '.' = pellet, 'O' = power pellet (energizer)
# 'D' = ghost door, 'G' = ghost start, 'P' = pacman start, ' ' = empty
# '-' = ghost house wall (ghosts can pass, pacman cannot)
MAZE_LAYOUT = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#O####.#####.##.#####.####O#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##### ## #####.#     ",
    "     #.##          ##.#     ",
    "     #.## ###DD### ##.#     ",
    "######.## #GGGGGG# ##.######",
    "      .   #GGGGGG#   .      ",
    "######.## #GGGGGG# ##.######",
    "     #.## ######## ##.#     ",
    "     #.##          ##.#     ",
    "     #.## ######## ##.#     ",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#O..##.......P .......##..O#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
]


@dataclass
class CurriculumStage:
    name: str
    num_ghosts: int
    has_power_pellets: bool
    max_steps: int
    win_threshold: float  # Pellet ratio to advance
    eval_episodes: int = 20


# Same 4-stage curriculum, all on classic arcade maze
CURRICULUM = [
    CurriculumStage("stage1_forage_only", num_ghosts=0, has_power_pellets=False,
                     max_steps=800, win_threshold=0.50),
    CurriculumStage("stage2_one_ghost", num_ghosts=1, has_power_pellets=False,
                     max_steps=1000, win_threshold=0.25),
    CurriculumStage("stage3_ghosts_power", num_ghosts=2, has_power_pellets=True,
                     max_steps=1200, win_threshold=0.20),
    CurriculumStage("stage4_full_game", num_ghosts=4, has_power_pellets=True,
                     max_steps=1500, win_threshold=0.15),
]


@dataclass
class TrainConfig:
    # Environment
    num_envs: int = 8
    frightened_duration: int = 42  # ~14 seconds at 3 fps game speed
    ghost_flash_duration: int = 6  # 2 seconds flashing before normal
    ghost_release_interval: int = 60
    scared_ghost_speed: float = 0.5  # Half speed when frightened

    # Model - High level (Transformer planner)
    high_seq_len: int = 16
    high_hidden_dim: int = 64
    high_num_layers: int = 2
    high_num_heads: int = 4
    num_codes: int = 8
    code_embed_dim: int = 16

    # Model - Low level (MLP executor)
    low_seq_len: int = 1
    low_hidden_dim: int = 128
    low_num_layers: int = 0
    low_num_heads: int = 0

    # Hierarchical
    temporal_bottleneck_k: int = 5
    gumbel_temp_start: float = 2.0
    gumbel_temp_end: float = 0.5
    gumbel_anneal_steps: int = 2_000_000

    # PPO
    rollout_length: int = 1024
    ppo_epochs: int = 4
    mini_batch_size: int = 512
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02
    high_entropy_coef: float = 0.20  # Stronger diversity pressure
    max_grad_norm: float = 0.5
    dropout: float = 0.0

    # Training schedule
    seed: int = 0  # Will be overridden per run
    total_timesteps: int = 10_000_000
    eval_interval: int = 8192
    save_interval: int = 16384
    log_interval: int = 8192

    # === Juice-drop reward mapping (faithful to original game) ===
    reward_pellet: float = 2.0        # bean: 2 drops
    reward_energizer: float = 4.0     # energizer: 4 drops
    reward_eat_ghost: float = 8.0     # scared ghost: 8 drops
    reward_clear_all: float = 20.0    # extra-large bonus
    reward_death: float = -15.0       # 5s timeout penalty
    reward_game_over: float = -30.0
    reward_step: float = -0.01        # Time pressure
