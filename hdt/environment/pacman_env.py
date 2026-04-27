"""Classic arcade Pac-Man environment with juice-drop rewards."""
import numpy as np
from config import (
    WALL, EMPTY, PELLET, POWER_PELLET, GHOST_DOOR, GHOST_HOUSE,
    MAZE_LAYOUT, ACTION_DELTAS, FEATURE_DIM,
    CHASE, SCATTER, FRIGHTENED, CurriculumStage, TrainConfig,
    UP, DOWN, LEFT, RIGHT,
)


class Ghost:
    def __init__(self, row, col, home_corner, ghost_id):
        self.start_row = row
        self.start_col = col
        self.row = row
        self.col = col
        self.home_corner = home_corner
        self.ghost_id = ghost_id
        self.mode = CHASE
        self.frightened_timer = 0
        self.active = False
        self.direction = UP
        self.eaten = False
        self.move_cooldown = 0  # For half-speed frightened movement

    def reset(self):
        self.row = self.start_row
        self.col = self.start_col
        self.mode = CHASE
        self.frightened_timer = 0
        self.active = False
        self.direction = UP
        self.eaten = False
        self.move_cooldown = 0

    @property
    def pos(self):
        return (self.row, self.col)


class PacmanEnv:
    def __init__(self, config: TrainConfig = None, stage: CurriculumStage = None):
        self.config = config or TrainConfig()
        self.stage = stage
        self._parse_maze()
        self.reset()

    def set_stage(self, stage: CurriculumStage):
        self.stage = stage
        self.reset()

    def _parse_maze(self):
        self.rows = len(MAZE_LAYOUT)
        self.cols = len(MAZE_LAYOUT[0])
        self.initial_grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.ghost_starts = []
        self.pacman_start = None
        self.door_positions = []
        self.power_pellet_positions = []
        self.tunnel_rows = set()  # Rows with open sides (tunnels)

        for r, row_str in enumerate(MAZE_LAYOUT):
            for c, ch in enumerate(row_str):
                if ch == '#' or ch == '-':
                    self.initial_grid[r, c] = WALL
                elif ch == '.':
                    self.initial_grid[r, c] = PELLET
                elif ch == 'O':
                    self.initial_grid[r, c] = POWER_PELLET
                    self.power_pellet_positions.append((r, c))
                elif ch == 'D':
                    self.initial_grid[r, c] = GHOST_DOOR
                    self.door_positions.append((r, c))
                elif ch == 'G':
                    self.initial_grid[r, c] = GHOST_HOUSE
                    self.ghost_starts.append((r, c))
                elif ch == 'P':
                    self.initial_grid[r, c] = EMPTY
                    self.pacman_start = (r, c)
                elif ch == ' ':
                    self.initial_grid[r, c] = EMPTY
            # Detect tunnel rows (open on sides)
            if row_str[0] == ' ' or row_str[-1] == ' ':
                self.tunnel_rows.add(r)

        # Ghost home corners
        corners = [
            (1, 1), (1, self.cols - 2),
            (self.rows - 2, 1), (self.rows - 2, self.cols - 2),
        ]
        # Deduplicate ghost starts and ensure 4 positions
        unique_starts = list(dict.fromkeys(self.ghost_starts))
        while len(unique_starts) < 4:
            unique_starts.append(unique_starts[0] if unique_starts
                                 else (self.rows // 2, self.cols // 2))
        self.ghosts_template = []
        for i in range(4):
            g = Ghost(unique_starts[i % len(unique_starts)][0],
                      unique_starts[i % len(unique_starts)][1],
                      corners[i], i)
            self.ghosts_template.append(g)

    def _count_pellets(self):
        return int(np.sum((self.grid == PELLET) | (self.grid == POWER_PELLET)))

    def reset(self):
        self.grid = self.initial_grid.copy()

        if self.stage and not self.stage.has_power_pellets:
            for r, c in self.power_pellet_positions:
                if self.grid[r, c] == POWER_PELLET:
                    self.grid[r, c] = PELLET

        self.pac_row, self.pac_col = self.pacman_start
        self.score = 0
        self.step_count = 0
        self.pellets_eaten = 0
        self.done = False
        self.lives = 3
        self.total_pellets = self._count_pellets()

        num_ghosts = 4
        if self.stage:
            num_ghosts = self.stage.num_ghosts

        self.ghosts = []
        for i in range(num_ghosts):
            gt = self.ghosts_template[i]
            g = Ghost(gt.start_row, gt.start_col, gt.home_corner, gt.ghost_id)
            self.ghosts.append(g)

        if self.ghosts:
            self.ghosts[0].active = True

        self.max_steps = self.stage.max_steps if self.stage else 1500
        return self.get_features()

    def _is_walkable(self, r, c, is_ghost=False):
        # Handle tunnel wrapping
        if c < 0:
            c = self.cols - 1
        elif c >= self.cols:
            c = 0
        if r < 0 or r >= self.rows:
            return False
        cell = self.grid[r, c]
        if cell == WALL:
            return False
        if not is_ghost and cell in (GHOST_DOOR, GHOST_HOUSE):
            return False
        return True

    def _wrap_col(self, c):
        """Handle tunnel wrapping for columns."""
        if c < 0:
            return self.cols - 1
        elif c >= self.cols:
            return 0
        return c

    def _move_ghosts(self):
        for ghost in self.ghosts:
            if not ghost.active or ghost.eaten:
                continue
            if ghost.mode == FRIGHTENED:
                ghost.frightened_timer -= 1
                if ghost.frightened_timer <= 0:
                    ghost.mode = CHASE
                # Half speed: skip every other step
                ghost.move_cooldown += 1
                if ghost.move_cooldown % 2 == 0:
                    continue  # Skip this move

            if ghost.mode == FRIGHTENED:
                self._move_ghost_random(ghost)
            elif ghost.mode == SCATTER:
                self._move_ghost_toward(ghost, ghost.home_corner)
            else:
                self._move_ghost_toward(ghost, (self.pac_row, self.pac_col))

    def _move_ghost_toward(self, ghost, target):
        best_action = None
        best_dist = float('inf')
        reverse = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

        for action in range(4):
            dr, dc = ACTION_DELTAS[action]
            nr = ghost.row + dr
            nc = self._wrap_col(ghost.col + dc)
            if not self._is_walkable(nr, nc, is_ghost=True):
                continue
            if action == reverse.get(ghost.direction):
                continue
            dist = abs(nr - target[0]) + abs(nc - target[1])
            if dist < best_dist:
                best_dist = dist
                best_action = (action, nr, nc)

        if best_action is None:
            for action in range(4):
                dr, dc = ACTION_DELTAS[action]
                nr = ghost.row + dr
                nc = self._wrap_col(ghost.col + dc)
                if self._is_walkable(nr, nc, is_ghost=True):
                    best_action = (action, nr, nc)
                    break

        if best_action:
            ghost.direction = best_action[0]
            ghost.row = best_action[1]
            ghost.col = best_action[2]

    def _move_ghost_random(self, ghost):
        valid = []
        reverse = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        for action in range(4):
            dr, dc = ACTION_DELTAS[action]
            nr = ghost.row + dr
            nc = self._wrap_col(ghost.col + dc)
            if self._is_walkable(nr, nc, is_ghost=True):
                if action != reverse.get(ghost.direction):
                    valid.append((action, nr, nc))
        if not valid:
            for action in range(4):
                dr, dc = ACTION_DELTAS[action]
                nr = ghost.row + dr
                nc = self._wrap_col(ghost.col + dc)
                if self._is_walkable(nr, nc, is_ghost=True):
                    valid.append((action, nr, nc))
        if valid:
            choice = valid[np.random.randint(len(valid))]
            ghost.direction = choice[0]
            ghost.row = choice[1]
            ghost.col = choice[2]

    def _release_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            if not ghost.active and not ghost.eaten:
                if self.step_count >= (i * self.config.ghost_release_interval):
                    ghost.active = True
                    if self.door_positions:
                        ghost.row = self.door_positions[0][0] - 1
                        ghost.col = self.door_positions[0][1]

    def _check_ghost_collision(self):
        pac_pos = (self.pac_row, self.pac_col)
        for ghost in self.ghosts:
            if not ghost.active or ghost.eaten:
                continue
            if ghost.pos == pac_pos:
                if ghost.mode == FRIGHTENED:
                    ghost.eaten = True
                    ghost.row = ghost.start_row
                    ghost.col = ghost.start_col
                    ghost.active = False
                    ghost.mode = CHASE
                    ghost.frightened_timer = 0
                    self.score += 200
                    return "eat_ghost"
                else:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.done = True
                        return "game_over"
                    else:
                        self.pac_row, self.pac_col = self.pacman_start
                        for g in self.ghosts:
                            g.reset()
                        if self.ghosts:
                            self.ghosts[0].active = True
                        return "death"
        return "none"

    def step(self, action: int):
        if self.done:
            return self.get_features(), 0.0, True, {}

        self.step_count += 1
        cfg = self.config
        reward = cfg.reward_step

        # Move pacman (with tunnel wrapping)
        dr, dc = ACTION_DELTAS[action]
        nr = self.pac_row + dr
        nc = self._wrap_col(self.pac_col + dc)
        if self._is_walkable(nr, nc):
            self.pac_row, self.pac_col = nr, nc

        # Check cell
        cell = self.grid[self.pac_row, self.pac_col]
        if cell == PELLET:
            self.grid[self.pac_row, self.pac_col] = EMPTY
            self.score += 10
            self.pellets_eaten += 1
            reward = cfg.reward_pellet  # 2 drops
        elif cell == POWER_PELLET:
            self.grid[self.pac_row, self.pac_col] = EMPTY
            self.score += 50
            self.pellets_eaten += 1
            reward = cfg.reward_energizer  # 4 drops
            for ghost in self.ghosts:
                if ghost.active and not ghost.eaten:
                    ghost.mode = FRIGHTENED
                    ghost.frightened_timer = cfg.frightened_duration
                    ghost.move_cooldown = 0

        # Ghost collisions
        collision = self._check_ghost_collision()
        if collision == "eat_ghost":
            reward = cfg.reward_eat_ghost  # 8 drops
        elif collision == "game_over":
            reward = cfg.reward_game_over
        elif collision == "death":
            reward = cfg.reward_death

        if not self.done:
            self._release_ghosts()
            self._move_ghosts()
            collision2 = self._check_ghost_collision()
            if collision2 == "eat_ghost":
                reward = cfg.reward_eat_ghost
            elif collision2 == "game_over":
                reward = cfg.reward_game_over
            elif collision2 == "death":
                reward = cfg.reward_death
            collision = collision2 if collision2 != "none" else collision

        # Win
        if self.pellets_eaten >= self.total_pellets:
            self.done = True
            reward += cfg.reward_clear_all  # Extra-large bonus

        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True

        info = {
            "score": self.score,
            "pellets_eaten": self.pellets_eaten,
            "total_pellets": self.total_pellets,
            "lives": self.lives,
            "collision": collision,
            "win": self.pellets_eaten >= self.total_pellets,
        }
        return self.get_features(), reward, self.done, info

    def get_features(self) -> np.ndarray:
        """42-dim feature vector."""
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        pr, pc = self.pac_row, self.pac_col
        max_dist = self.rows + self.cols

        features[0] = pr / max(self.rows - 1, 1)
        features[1] = pc / max(self.cols - 1, 1)

        for i in range(4):
            base = 2 + i * 5
            if i < len(self.ghosts):
                ghost = self.ghosts[i]
                if ghost.active and not ghost.eaten:
                    features[base] = (ghost.row - pr) / max_dist
                    features[base + 1] = (ghost.col - pc) / max_dist
                    features[base + 2] = (abs(ghost.row - pr) + abs(ghost.col - pc)) / max_dist
                    features[base + 3] = 1.0 if ghost.mode == FRIGHTENED else 0.0
                    features[base + 4] = 1.0

        pellet_positions = list(zip(*np.where(self.grid == PELLET)))
        if pellet_positions:
            dists = [abs(r - pr) + abs(c - pc) for r, c in pellet_positions]
            idx = np.argmin(dists)
            nr, nc = pellet_positions[idx]
            features[22] = (nr - pr) / max_dist
            features[23] = (nc - pc) / max_dist
            features[24] = dists[idx] / max_dist

        pp_positions = list(zip(*np.where(self.grid == POWER_PELLET)))
        if pp_positions:
            dists = [abs(r - pr) + abs(c - pc) for r, c in pp_positions]
            idx = np.argmin(dists)
            nr, nc = pp_positions[idx]
            features[25] = (nr - pr) / max_dist
            features[26] = (nc - pc) / max_dist
            features[27] = dists[idx] / max_dist

        remaining = np.sum((self.grid == PELLET) | (self.grid == POWER_PELLET))
        features[28] = remaining / max(self.total_pellets, 1)

        max_fright = self.config.frightened_duration
        active_fright = max(
            (g.frightened_timer for g in self.ghosts if g.active), default=0
        )
        features[29] = active_fright / max(max_fright, 1)

        for i in range(4):
            dr, dc = ACTION_DELTAS[i]
            nr = pr + dr
            nc = self._wrap_col(pc + dc)
            features[30 + i] = 1.0 if self._is_walkable(nr, nc) else 0.0

        for i in range(4):
            dr, dc = ACTION_DELTAS[i]
            nr = pr + dr
            nc = self._wrap_col(pc + dc)
            if self._is_walkable(nr, nc):
                cell = self.grid[nr, nc]
                features[34 + i] = 1.0 if cell in (PELLET, POWER_PELLET) else 0.0

        look_ahead = 8
        for i in range(4):
            dr, dc = ACTION_DELTAS[i]
            count = 0
            r, c = pr, pc
            for _ in range(look_ahead):
                r = r + dr
                c = self._wrap_col(c + dc)
                if not self._is_walkable(r, c):
                    break
                if self.grid[r, c] in (PELLET, POWER_PELLET):
                    count += 1
            features[38 + i] = count / look_ahead

        return features

    def render_ascii(self) -> str:
        display = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == (self.pac_row, self.pac_col):
                    row.append('C')
                elif any(g.pos == (r, c) and g.active and not g.eaten
                         for g in self.ghosts):
                    ghost = next(g for g in self.ghosts
                                 if g.pos == (r, c) and g.active and not g.eaten)
                    if ghost.mode == FRIGHTENED:
                        if ghost.frightened_timer <= self.config.ghost_flash_duration:
                            row.append('f')  # Flashing
                        else:
                            row.append('F')  # Frightened
                    else:
                        row.append(str(ghost.ghost_id))
                else:
                    cell = self.grid[r, c]
                    chars = {WALL: '#', PELLET: '.', POWER_PELLET: 'O',
                             GHOST_DOOR: 'D', GHOST_HOUSE: '_', EMPTY: ' '}
                    row.append(chars.get(cell, ' '))
            display.append(''.join(row))
        return '\n'.join(display)
